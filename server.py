"""Local chat HTTP adapter for the bundled ExLlamaV3 runtime (text, tools, optional image input)."""
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import argparse
import os
# 加速開關要在載入 exllamav3 之前寫進環境變數，所以先從命令列取值（argparse 也有登記同名參數）
def _pre_flag(name):
    for i, a in enumerate(sys.argv):
        if a == name:
            return sys.argv[i + 1] if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith('-') else '1'
        if a.startswith(name + '='):
            return a.split('=', 1)[1]
    return None
import sys
# RDNA4 多列 WMMA GEMV（8～64 列）：server 預設開、64 列
os.environ['EXL3_GEMV_WMMA'] = _pre_flag('--gemv-wmma') or os.environ.get('EXL3_GEMV_WMMA', '1')
os.environ['EXL3_GEMV_MAX_M'] = _pre_flag('--gemv-max-m') or os.environ.get('EXL3_GEMV_MAX_M', '64')
if _pre_flag('--gemv-wmma-tn'):
    os.environ['EXL3_GEMV_WMMA_TN'] = _pre_flag('--gemv-wmma-tn')
if _pre_flag('--dflash2-host-cb'):
    os.environ['EXL3_DFLASH2_HOST_CB'] = '1'
if _pre_flag('--qc-staging-sliced'):
    os.environ['EXL3_QC_STAGING_SLICED'] = '1'
import base64
import binascii
from collections import OrderedDict
import copy
from datetime import datetime
import hashlib
import io
import json
import math
import queue
import re
import select
import socket
import sys
import threading
import time
import traceback
from urllib.parse import urlsplit
import uuid

ROOT = Path(__file__).resolve().parent
UI_DIR = ROOT / 'web/llama-ui'


def load_ui_manifest():
    """llama.cpp 內建網頁（從 GGUFRun 的 llama-server b11642 取出）：路徑 → 檔案／Content-Type／是否 gzip。"""
    try:
        files = json.loads((UI_DIR / 'manifest.json').read_text(encoding='utf-8'))['files']
    except (OSError, ValueError, KeyError):
        return {}
    return {p: m for p, m in files.items() if (UI_DIR / m['file']).is_file()}


UI_FILES = load_ui_manifest()
LOG_LOCK = threading.Lock()


def log(line):
    """多執行緒同時寫 log 時整行一次寫出，不會互相插字。"""
    with LOG_LOCK:
        sys.stdout.write(str(line).rstrip('\n') + '\n')
        sys.stdout.flush()


def speed_line(t, cancelled=False, note=''):
    """每次請求結束印一行，格式比照 llama-server 的 prompt eval／eval 統計。"""
    head = '[EXL3] 已取消' + (f'（{note}）' if note else '') if cancelled else '[EXL3] 完成'
    out = [f"{head}：預填 {t['prompt_n']} tok／{t['prompt_ms'] / 1000:.2f} s"
           + (f"（{t['prompt_per_second']:.1f} tok/s）" if t['prompt_n'] and t['prompt_ms'] else ''),
           f"快取重用 {t['cache_n']} tok",
           f"生成 {t['predicted_n']} tok／{t['predicted_ms'] / 1000:.2f} s"
           + (f"（{t['predicted_per_second']:.1f} tok/s）" if t['predicted_n'] and t['predicted_ms'] else '')]
    if t.get('draft_n'):
        out.append(f"草稿接受 {t['draft_n_accepted']}/{t['draft_n']}（{t['draft_n_accepted'] / t['draft_n']:.0%}）")
    return '｜'.join(out)


def prompt_messages(messages):
    if not isinstance(messages, list) or not messages:
        raise ValueError('messages 必須是非空陣列')
    system, pairs, pending = [], [], None
    for msg in messages:
        if not isinstance(msg, dict) or not isinstance(msg.get('content'), str):
            raise ValueError('目前文字聊天的 content 必須是字串')
        role, text = msg.get('role'), msg['content']
        if role == 'system':
            system.append(text)
        elif role == 'user':
            pending = text if pending is None else pending + '\n\n' + text
        elif role == 'assistant' and pending is not None:
            pairs.append((pending, text))
            pending = None
        else:
            raise ValueError('支援 system、user、assistant 文字訊息；對話須以 user 結尾')
    if pending is None:
        raise ValueError('對話須以 user 結尾')
    pairs.append((pending, None))
    return '\n\n'.join(system), pairs


def load_hf_template(model_dir):
    d = Path(model_dir)
    p = d / 'chat_template.jinja'
    if p.is_file():
        return p.read_text(encoding='utf-8')
    try:
        cfg = json.loads((d / 'tokenizer_config.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    t = cfg.get('chat_template')
    if isinstance(t, list):
        t = next((x.get('template') for x in t if isinstance(x, dict) and x.get('name') == 'default'), None)
    return t if isinstance(t, str) and t.strip() else None


EFFORT_LEVELS = ('minimal', 'low', 'medium', 'high', 'xhigh', 'max', 'ultra')


def template_reasoning_efforts(tree):
    """讀取模板明列的 reasoning_effort 集合，不推測模型名稱或更改模板。"""
    from jinja2 import nodes
    names = {'reasoning_effort'}
    assignments = list(tree.find_all(nodes.Assign))
    changed = True
    while changed:
        changed = False
        for assignment in assignments:
            target, value = assignment.target, assignment.node
            refs = ([value] if isinstance(value, nodes.Name) else []) + list(value.find_all(nodes.Name))
            if isinstance(target, nodes.Name) and target.name not in names and any(n.name in names for n in refs):
                names.add(target.name)
                changed = True
    declared = []
    for comparison in tree.find_all(nodes.Compare):
        if not isinstance(comparison.expr, nodes.Name) or comparison.expr.name not in names:
            continue
        if len(comparison.ops) != 1:
            continue
        operand = comparison.ops[0]
        if operand.op not in ('in', 'notin') or not isinstance(operand.expr, (nodes.Tuple, nodes.List)):
            continue
        items = operand.expr.items
        if items and all(isinstance(item, nodes.Const) and isinstance(item.value, str) for item in items):
            declared.append(tuple(item.value for item in items))
    return declared[0] if declared and all(values == declared[0] for values in declared) else ()


def resolve_reasoning(body, kwargs, default_effort, default_thinking, supported=()):
    """每次請求重新解析；客戶端設定優先，只有未支援的標準層級才做對應。"""
    think = body.get('enable_thinking', kwargs.get('enable_thinking', default_thinking))
    if not isinstance(think, bool):
        raise ValueError('enable_thinking 必須是布林值')
    effort = body.get('reasoning_effort')
    if effort is None:
        effort = kwargs.get('reasoning_effort')
    requested = effort
    if effort is None or (isinstance(effort, str) and effort.strip().lower() in ('', 'default', 'auto')):
        effort = default_effort
    if effort is not None:
        if not isinstance(effort, str):
            raise ValueError('reasoning_effort 必須是字串')
        effort = effort.strip()
        if effort.lower() in ('', 'default', 'auto'):
            effort = None
        elif effort.lower() in ('none', 'off', 'disabled', 'false'):
            think, effort = False, None
        elif effort.lower() in EFFORT_LEVELS:
            effort = effort.lower()
    if not think:
        return False, None, requested
    if supported and effort is not None and effort not in supported:
        available = [level for level in EFFORT_LEVELS if level in supported]
        if effort in EFFORT_LEVELS and available:
            rank = EFFORT_LEVELS.index(effort)
            effort = next((level for level in available if EFFORT_LEVELS.index(level) >= rank), available[-1])
    return think, effort, requested


class HFTemplate:
    """Render a model's own chat_template with the same Jinja rules Hugging Face uses."""
    def __init__(self, source, bos, eos):
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        from jinja2.exceptions import TemplateError

        def raise_exception(message):
            raise TemplateError(message)

        def tojson(x, indent=None, separators=None, sort_keys=False, ensure_ascii=False):
            return json.dumps(x, indent=indent, separators=separators, sort_keys=sort_keys, ensure_ascii=ensure_ascii)

        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                            extensions=['jinja2.ext.loopcontrols'])
        env.filters['tojson'] = tojson
        env.globals['raise_exception'] = raise_exception
        env.globals['strftime_now'] = lambda fmt: datetime.now().strftime(fmt)
        tree = env.parse(source)
        self.reasoning_efforts = template_reasoning_efforts(tree)
        self.template = env.from_string(tree)
        self.bos, self.eos = bos or '', eos or ''

    def render(self, messages, think, extra=None, tools=None):
        values = dict(extra or {})
        values.update(messages=messages, add_generation_prompt=True, enable_thinking=think,
                      bos_token=self.bos, eos_token=self.eos, tools=tools)
        return self.template.render(**values)


def image_url(part):
    """OpenAI image part → URL string（image_url 可以是字串或 {url: ...}）；不是圖片段落回 None。"""
    if not isinstance(part, dict) or part.get('type') != 'image_url':
        return None
    url = part.get('image_url')
    url = url.get('url') if isinstance(url, dict) else url
    if not isinstance(url, str) or not url:
        raise ValueError('image_url 必須有 url')
    return url


def text_content(content, image=None):
    """OpenAI content: a string, or parts joined.  Text parts as is; image parts become image(url) — the
    embedding's placeholder text — when the server has a vision tower, otherwise they are rejected."""
    if content is None:
        return ''
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for part in content:
            url = image_url(part)
            if url is not None:
                if image is None:
                    raise ValueError('這個 EXL3 server 沒有開視覺（GGUFRun「視覺」選 GPU 或 RAM 後重啟），請只傳文字')
                texts.append(image(url))
                continue
            if not isinstance(part, dict) or part.get('type') != 'text' or not isinstance(part.get('text'), str):
                raise ValueError('EXL3 只支援文字與圖片段落（不支援音訊／影片）')
            texts.append(part['text'])
        return '\n\n'.join(texts)
    raise ValueError('content 必須是字串或段落陣列')


def decode_image(url):
    """data:image/...;base64,... → (bytes, PIL.Image)。只收內嵌圖片，不替網頁去抓外部網址。"""
    head, sep, data = url.partition(',')
    if not sep or not head.startswith('data:image/') or not head.endswith(';base64'):
        raise ValueError('圖片須是 data:image/...;base64 內嵌格式（不抓外部網址）')
    try:
        raw = base64.b64decode(data, validate=False)
    except (binascii.Error, ValueError):
        raise ValueError('圖片 base64 解碼失敗') from None
    from PIL import Image
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except Exception as exc:
        raise ValueError(f'無法讀取圖片：{exc}') from None
    return raw, img


def text_messages(messages):
    """OpenAI messages -> {role, content} with string content (the built-in formatters' input)."""
    if not isinstance(messages, list):
        return messages
    out = []
    for msg in messages:
        if not isinstance(msg, dict):
            out.append(msg)
            continue
        if msg.get('role') == 'tool' or msg.get('tool_calls'):
            raise ValueError('工具呼叫需要模型自帶的聊天模板（chat_template.jinja）')
        out.append({'role': msg.get('role'), 'content': text_content(msg.get('content'), no_template_image)})
    return out


def no_template_image(_url):
    raise ValueError('看圖需要模型自帶的聊天模板（chat_template.jinja）')


def template_call(call):
    """An assistant tool call (OpenAI: arguments as a JSON string) -> the template's form (arguments as an object)."""
    fn = call.get('function') if isinstance(call, dict) else None
    if not isinstance(fn, dict) or not isinstance(fn.get('name'), str) or not fn['name']:
        raise ValueError('tool_calls 每項須有 function.name')
    args = fn.get('arguments')
    if args is None or args == '':
        args = {}
    elif isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError as exc:
            raise ValueError(f'tool_calls 的 arguments 不是合法 JSON：{exc}') from exc
    if not isinstance(args, dict):
        raise ValueError('tool_calls 的 arguments 必須是 JSON object')
    return {'type': 'function', 'function': {'name': fn['name'], 'arguments': args}}


def template_messages(messages, image=None):
    """OpenAI messages -> the model's chat template input: text content, tool results, the assistant's tool calls and
    the reasoning of earlier turns (so the history renders as the model wrote it); consecutive user turns merged.
    image(url) turns a user's image part into the embedding placeholder text (None: images rejected)."""
    if not isinstance(messages, list) or not messages:
        raise ValueError('messages 必須是非空陣列')
    out = []
    for msg in messages:
        if not isinstance(msg, dict):
            raise ValueError('messages 每項須是物件')
        role = msg.get('role')
        if role not in ('system', 'user', 'assistant', 'tool'):
            raise ValueError('支援 system、user、assistant、tool 訊息')
        m = {'role': role, 'content': text_content(msg.get('content'), image if role == 'user' else None)}
        if role == 'assistant':
            if isinstance(msg.get('reasoning_content'), str) and msg['reasoning_content']:
                m['reasoning_content'] = msg['reasoning_content']
            calls = msg.get('tool_calls')
            if calls:
                if not isinstance(calls, list):
                    raise ValueError('tool_calls 必須是陣列')
                m['tool_calls'] = [template_call(c) for c in calls]
        elif role == 'tool' and isinstance(msg.get('tool_call_id'), str):
            m['tool_call_id'] = msg['tool_call_id']
        if out and role == 'user' and out[-1]['role'] == 'user':
            out[-1]['content'] += '\n\n' + m['content']
        else:
            out.append(m)
    if out[-1]['role'] not in ('user', 'tool'):
        raise ValueError('對話須以 user 或 tool 訊息結尾')
    return out


def request_tools(tools, tool_choice):
    """OpenAI tools -> the list the chat template renders (None: no tools, also for tool_choice "none")."""
    if tools in (None, []):
        return None
    if not isinstance(tools, list):
        raise ValueError('tools 必須是陣列')
    out = []
    for t in tools:
        fn = t.get('function') if isinstance(t, dict) and t.get('type', 'function') == 'function' else None
        if not isinstance(fn, dict) or not isinstance(fn.get('name'), str) or not fn['name']:
            raise ValueError('tools 每項須是 {"type": "function", "function": {"name": ...}}')
        out.append({'type': 'function', 'function': fn})
    if tool_choice == 'none':
        return None
    if tool_choice not in (None, 'auto', 'required') and not isinstance(tool_choice, dict):
        raise ValueError('tool_choice 必須是 none、auto、required 或指定函式')
    return out


def tool_value(raw, schema):
    """A <parameter> value (text) -> the JSON type its schema names; unknown types: JSON if it parses, else text."""
    if raw.startswith('\n'):
        raw = raw[1:]
    if raw.endswith('\n'):
        raw = raw[:-1]
    kind = schema.get('type') if isinstance(schema, dict) else None
    if isinstance(kind, list):
        kind = next((k for k in kind if k != 'null'), None)
    if kind == 'string':
        return raw
    text = raw.strip()
    if text.lower() == 'null':
        return None
    try:
        if kind == 'integer':
            return int(text)
        if kind == 'number':
            f = float(text)
            return int(f) if f.is_integer() and not any(c in text for c in '.eE') else f
        if kind == 'boolean':
            if text.lower() in ('true', 'false'):
                return text.lower() == 'true'
            return raw
        return json.loads(text)
    except ValueError:
        return raw


class ToolCallParser:
    """Content text -> ('content', text) and ('tool_call', OpenAI tool call) pieces.  Qwen3.5-family XML:
    <tool_call>\n<function=NAME>\n<parameter=P>\nVALUE\n</parameter>\n</function>\n</tool_call>.
    A call is sent once it is complete; text that only might start a call is held back."""
    OPEN, CLOSE = '<tool_call>', '</tool_call>'
    FUNC = re.compile(r'<function=([^>\n]+)>(.*?)(?:</function>|$)', re.S)
    PARAM = re.compile(r'<parameter=([^>\n]+)>(.*?)(?:</parameter>|(?=<parameter=)|$)', re.S)

    def __init__(self, tools):
        self.schemas = {}
        for t in tools:
            fn = t['function']
            params = fn.get('parameters') if isinstance(fn.get('parameters'), dict) else {}
            props = params.get('properties') if isinstance(params.get('properties'), dict) else {}
            self.schemas[fn['name']] = props
        self.buffer, self.inside, self.calls = '', False, 0

    def _call(self, body, closed):
        found = self.FUNC.search(body)
        if not found:
            return [('content', self.OPEN + body + (self.CLOSE if closed else ''))]
        name = found.group(1).strip()
        props = self.schemas.get(name, {})
        args = {}
        for p in self.PARAM.finditer(found.group(2)):
            key = p.group(1).strip()
            args[key] = tool_value(p.group(2), props.get(key))
        call = {'index': self.calls, 'id': 'call_' + uuid.uuid4().hex[:24], 'type': 'function',
                'function': {'name': name, 'arguments': json.dumps(args, ensure_ascii=False)}}
        self.calls += 1
        return [('tool_call', call)]

    def feed(self, text, final=False):
        self.buffer += text
        out = []
        while self.buffer:
            if self.inside:
                i = self.buffer.find(self.CLOSE)
                if i < 0:
                    if final:
                        out += self._call(self.buffer, False)
                        self.buffer = ''
                    break
                out += self._call(self.buffer[:i], True)
                self.buffer = self.buffer[i + len(self.CLOSE):]
                self.inside = False
                continue
            i = self.buffer.find(self.OPEN)
            if i >= 0:
                head = self.buffer[:i].rstrip()
                if head.strip():
                    out.append(('content', head))
                self.buffer = self.buffer[i + len(self.OPEN):]
                self.inside = True
                continue
            if final:
                rest = self.buffer if not self.calls else self.buffer.strip()
                if rest.strip():
                    out.append(('content', rest))
                self.buffer = ''
                break
            # hold back what could still become "<tool_call>", and trailing whitespace before it
            keep = 0
            for n in range(1, min(len(self.OPEN), len(self.buffer) + 1)):
                if self.buffer.endswith(self.OPEN[:n]):
                    keep = n
            cut = len(self.buffer) - keep
            body = self.buffer[:cut]
            emit = body.rstrip()
            if self.calls and not emit.strip():
                emit = ''
            if emit:
                out.append(('content', emit))
            self.buffer = body[len(emit):] + self.buffer[cut:] if emit else self.buffer
            break
        return out


def chat_messages(messages):
    """Validate OpenAI-style text messages and merge consecutive user turns."""
    prompt_messages(messages)
    result = []
    for msg in messages:
        role, text = msg['role'], msg['content']
        if result and role == 'user' and result[-1]['role'] == 'user':
            result[-1]['content'] += '\n\n' + text
        else:
            result.append({'role': role, 'content': text})
    return result


class ReasoningParser:
    """Keep partial delimiters across chunks; expose reasoning separately."""
    def __init__(self, implicit=False):
        self.buffer = ''
        self.reasoning = implicit
        self.trim = False

    def feed(self, text, final=False):
        self.buffer += text
        result = []
        while self.buffer:
            tag = '</think>' if self.reasoning else '<think>'
            i = self.buffer.find(tag)
            if i >= 0:
                if i:
                    part = self.buffer[:i]
                    if self.trim and not self.reasoning:
                        part = part.lstrip('\r\n')
                        self.trim = not part
                    if part:
                        result.append(('reasoning_content' if self.reasoning else 'content', part))
                self.buffer = self.buffer[i + len(tag):]
                self.reasoning = not self.reasoning
                self.trim = not self.reasoning
                continue
            keep = 0
            if not final:
                for n in range(1, min(len(tag), len(self.buffer) + 1)):
                    if self.buffer.endswith(tag[:n]):
                        keep = n
            emit = self.buffer[:-keep] if keep else self.buffer
            self.buffer = self.buffer[-keep:] if keep else ''
            if emit and self.trim and not self.reasoning:
                emit = emit.lstrip('\r\n')
                self.trim = not emit
            if emit:
                result.append(('reasoning_content' if self.reasoning else 'content', emit))
            break
        return result


class LiveTimings:
    """llama-server 格式的 timings／prompt_progress（llama.cpp 網頁靠這兩個欄位顯示 tok/s 與預填進度）。
    生成中讀 ExLlamaV3 Job 的即時計數；結束時改用 eos 結果的正式數字。"""
    def __init__(self, job, prompt_count, page_size=256):
        self.job, self.prompt_count, self.page_size = job, prompt_count, page_size

    def cached(self, final=None):
        if final and final.get('cached_tokens') is not None:
            return int(final['cached_tokens'])
        job = self.job
        return int(getattr(job, 'cached_pages', 0) or 0) * self.page_size + int(getattr(job, 'cached_tokens', 0) or 0)

    def timings(self, final=None):
        job, now = self.job, time.time()
        if final:
            cache_n = self.cached(final)
            prompt_ms = (final.get('time_prefill') or 0) * 1000
            predicted_n = int(final.get('new_tokens') or 0)
            predicted_ms = (final.get('time_generate') or 0) * 1000
            acc, rej = final.get('accepted_draft_tokens'), final.get('rejected_draft_tokens')
        else:
            first_prefill, first_token = getattr(job, 'time_first_prefill', None), getattr(job, 'time_first_token', None)
            cache_n = self.cached()
            prompt_ms = ((first_token or now) - first_prefill) * 1000 if first_prefill else 0
            predicted_n = max(0, int(getattr(job, 'new_tokens', 0) or 0))
            predicted_ms = (now - first_token) * 1000 if first_token else 0
            acc, rej = getattr(job, 'accepted_draft_tokens', None), getattr(job, 'rejected_draft_tokens', None)
        prompt_n = max(0, self.prompt_count - cache_n)
        out = {'cache_n': cache_n, 'prompt_n': prompt_n, 'prompt_ms': prompt_ms,
               'prompt_per_token_ms': prompt_ms / prompt_n if prompt_n else 0,
               'prompt_per_second': prompt_n * 1000 / prompt_ms if prompt_ms else 0,
               'predicted_n': predicted_n, 'predicted_ms': predicted_ms,
               'predicted_per_token_ms': predicted_ms / predicted_n if predicted_n else 0,
               'predicted_per_second': predicted_n * 1000 / predicted_ms if predicted_ms else 0}
        if isinstance(acc, int) and isinstance(rej, int):
            out.update(draft_n=acc + rej, draft_n_accepted=acc)
        return out

    def progress(self, result):
        """stage == 'prefill' 的結果 → prompt_progress；processed／total 都含已快取的前文。"""
        first = getattr(self.job, 'time_first_prefill', None)
        return {'total': int(result.get('max_progress') or self.prompt_count), 'cache': self.cached(),
                'processed': int(result.get('curr_progress') or 0),
                'time_ms': (time.time() - first) * 1000 if first else 0}


class ProgressLog:
    """生成途中每隔幾秒印一行進度，比照 llama-server 的 prompt processing／n_gen、tg、tg_3s。"""
    def __init__(self, live, interval=3.0, clock=time.time, request_id=None):
        self.live, self.interval, self.clock = live, interval, clock
        self.prefix = f'[{request_id}] ' if request_id else ''
        self.last, self.mark = clock(), None

    def prefill(self, result):
        now = self.clock()
        if now - self.last < self.interval:
            return
        self.last = now
        p = self.live.progress(result)
        done, todo, sec = max(0, p['processed'] - p['cache']), max(0, p['total'] - p['cache']), p['time_ms'] / 1000
        log(self.prefix + f'[EXL3] 預填 {done}/{todo} tok' + (f'（{done / todo:.0%}）' if todo else '') + f'｜{sec:.1f} s'
            + (f'｜{done / sec:.1f} tok/s' if sec else ''))

    def generate(self):
        now, t = self.clock(), self.live.timings()
        n = t['predicted_n']
        if self.mark is None:                    # 第一個 token：開始計時，3 秒後才印
            self.mark, self.last = (now, n), now
            return
        if now - self.last < self.interval:
            return
        recent = (n - self.mark[1]) / (now - self.mark[0]) if now > self.mark[0] else 0
        out = [f'[EXL3] 生成 {n} tok', f"平均 {t['predicted_per_second']:.1f} tok/s",
               f'近 {now - self.mark[0]:.0f} 秒 {recent:.1f} tok/s']
        if t.get('draft_n'):
            out.append(f"草稿接受 {t['draft_n_accepted']}/{t['draft_n']}（{t['draft_n_accepted'] / t['draft_n']:.0%}）")
        log(self.prefix + '｜'.join(out))
        self.mark, self.last = (now, n), now


def response_budget(prompt, context, requested, draft):
    if requested is None or requested == -1:          # -1：llama.cpp 的「不限」，同 0
        requested = 0
    if isinstance(requested, bool) or not isinstance(requested, int) or requested < 0:
        raise ValueError('max_tokens 必須是非負整數；0 或 -1 使用剩餘上下文')
    available = context - prompt - draft - 1
    if available < 1:
        raise ValueError(f'對話超出上下文容量：提示詞 {prompt}，上下文 {context}；請清空或縮短對話')
    return min(requested, available) if requested else available


class RequestState:
    """One HTTP request. Only the scheduler touches its Job or the shared GPU."""
    def __init__(self, body, slot):
        self.body, self.slot = body, slot
        self.rid = 'chatcmpl-' + uuid.uuid4().hex
        self.prepared, self.done, self.cancelled = threading.Event(), threading.Event(), threading.Event()
        self.results = queue.Queue(maxsize=256)
        self.job, self.serial, self.info, self.error = None, None, None, None


class AutoDraft:
    """--auto-draft：依實測每秒產出的 token 數，替目前同時生成的路數挑草稿數。
    檔位 0（暫停打草稿，草稿模型照樣跟著同步）～-ndt；每個路數各自量；換檔要贏 5% 以上，之後定期重試鄰近檔位。"""
    WARM, EMA, REPROBE, MARGIN = 6, 0.15, 400, 1.05

    def __init__(self, generator, ceiling):
        self.gen, self.ceiling = generator, ceiling
        self.cands = sorted({c for c in (0, 1, 2, 3, 4, 5, 7, ceiling) if 0 <= c <= ceiling})
        self.stats, self.cur, self.count, self.probe = {}, {}, {}, {}

    def pick(self, active):
        key = active
        self.count[key] = self.count.get(key, 0) + 1
        if self.probe.get(key):
            ndt, left = self.probe[key]
            self.probe[key] = (ndt, left - 1) if left > 1 else None
            return ndt
        for c in self.cands:
            if self.stats.get((key, c), (0, 0))[1] < self.WARM:
                return c
        cur = self.cur.get(key)
        best = max(self.cands, key=lambda c: self.stats[(key, c)][0])
        if cur is None or self.stats[(key, best)][0] > self.stats[(key, cur)][0] * self.MARGIN:
            if cur != best:
                rate = {c: round(self.stats[(key, c)][0], 1) for c in self.cands}
                log(f'[EXL3] 自動草稿｜{active} 路 → ' + (f'草稿 {best} token' if best else '暫停草稿') + f'（實測 tok/s：{rate}）')
            self.cur[key] = cur = best
        if self.count[key] % self.REPROBE == 0:
            i = self.cands.index(cur)
            near = [self.cands[j] for j in (i - 1, i + 1) if 0 <= j < len(self.cands)]
            if near:
                self.probe[key] = (near[(self.count[key] // self.REPROBE) % len(near)], 3)
        return cur

    def record(self, active, ndt, tokens, seconds, accepted=None):
        if tokens <= 0 or seconds <= 0:
            return
        rate = tokens / seconds
        old = self.stats.get((active, ndt))
        self.stats[(active, ndt)] = (rate, 1) if old is None else (old[0] + self.EMA * (rate - old[0]), old[1] + 1)


class AcceptDraft:
    """--auto-draft-accept：每輪依「最近各位置的草稿猜中率」與「各列數級距的每輪耗時」挑草稿數（0～-ndt）。
    猜中率跟內容走，換題幾輪內就跟上；耗時只看路數與列數級距（路數 ×（草稿數＋1）補到 8／16／32／64 列），跟內容無關。
    統計沿用驗證時本來就算出的每路接受數，只在 CPU 端做整數與少量浮點運算。"""
    DECAY, PRIOR, WEIGHT, WARM, EMA, PROBE, MARGIN = 0.95, 0.6, 2.0, 3, 0.2, 40, 1.03

    def __init__(self, generator, ceiling):
        self.gen, self.ceiling = generator, ceiling
        self.cands = list(range(ceiling + 1))
        self.hit = [0.0] * (ceiling + 2)
        self.tried = [0.0] * (ceiling + 2)
        self.cost, self.cur, self.count, self.probe = {}, {}, {}, {}

    @staticmethod
    def tier(rows):
        t = 8
        while t < rows:
            t *= 2
        return t

    def key(self, n, d):
        return n, self.tier(n * (d + 1)), d > 0

    def groups(self, n):
        rep = {}
        for d in self.cands:
            rep[self.key(n, d)] = d
        return rep

    def rates(self):
        q, prev = [], self.PRIOR
        for p in range(1, self.ceiling + 1):
            prev = (self.hit[p] + self.WEIGHT * prev) / (self.tried[p] + self.WEIGHT)
            q.append(prev)
        return q

    def expect(self, q, d):
        e, run = 1.0, 1.0
        for p in range(d):
            run *= q[p]
            e += run
        return e

    def pick(self, active):
        n = active
        self.count[n] = self.count.get(n, 0) + 1
        rep = self.groups(n)
        for k in sorted(rep, key=lambda k: -rep[k]):
            if self.cost.get(k, (0, 0))[1] < self.WARM:
                return rep[k]
        if self.probe.get(n) is not None:
            d, self.probe[n] = self.probe[n], None
            return d
        q = self.rates()
        score = {d: n * self.expect(q, d) / self.cost[self.key(n, d)][0] for d in self.cands}
        best = max(self.cands, key=lambda d: score[d])
        cur = self.cur.get(n)
        if cur is None or score[best] > score[cur] * self.MARGIN:
            if cur != best:
                log(f'[EXL3] 自動草稿（猜中率）｜{n} 路 → ' + (f'草稿 {best} token' if best else '暫停草稿')
                    + f'（各位置猜中率 {[round(v, 2) for v in q]}｜預估 tok/s {round(score[cur], 1) if cur is not None else "-"} → {round(score[best], 1)}）')
            self.cur[n] = cur = best
        if self.count[n] % self.PROBE == 0:
            above = sorted({rep[k] for k in rep if rep[k] > cur})
            if above:
                self.probe[n] = above[0]
        return cur

    def record(self, active, ndt, tokens, seconds, accepted=None):
        if seconds <= 0:
            return
        k = self.key(active, ndt)
        old = self.cost.get(k)
        self.cost[k] = (seconds, 1) if old is None else (old[0] + self.EMA * (seconds - old[0]), old[1] + 1)
        if ndt < 1 or not accepted:
            return
        for p in range(1, self.ceiling + 1):
            self.hit[p] *= self.DECAY
            self.tried[p] *= self.DECAY
        for a in accepted:
            a = max(0, min(int(a), ndt))
            for p in range(1, a + 1):
                self.hit[p] += 1
                self.tried[p] += 1
            if a < ndt:
                self.tried[a + 1] += 1


class GenerationScheduler:
    """Single GPU owner: continuously batch jobs and demultiplex results by serial.

    HTTP threads only reserve a slot, consume their own bounded result queue and set
    cancellation events. No HTTP thread calls prepare/enqueue/iterate/cancel.
    """
    def __init__(self, engine, parallel):
        self.engine, self.parallel = engine, parallel
        self.condition = threading.Condition()
        self.requests, self.by_serial = {}, {}
        self.stopping = False
        self.thread = threading.Thread(target=self.run, name='exl3-generation', daemon=True)
        self.thread.start()

    def submit(self, body):
        with self.condition:
            if self.stopping or not self.engine.ready:
                raise RuntimeError('生成服務尚未就緒')
            used = {r.slot for r in self.requests.values()}
            slot = next((s for s in range(self.parallel) if s not in used), None)
            if slot is None:
                return None
            state = RequestState(body, slot)
            self.requests[state.rid] = state
            self.condition.notify()
            return state

    def snapshot(self):
        with self.condition:
            return [{'id': r.slot, 'request_id': r.rid, 'is_processing': True}
                    for r in self.requests.values()]

    def cancel(self, rid):
        with self.condition:
            state = self.requests.get(rid)
            if state is None:
                return False
            state.cancelled.set()
            self.condition.notify()
            return True

    def close(self):
        with self.condition:
            self.stopping = True
            for state in self.requests.values():
                state.cancelled.set()
            self.condition.notify()
        self.thread.join(timeout=5)

    def finish(self, state, error=None, cancel=False):
        """Called only on the GPU owner thread; release GPU state before the slot."""
        try:
            if cancel and state.job is not None:
                self.engine.generator.cancel(state.job)
        finally:
            state.error = error
            if state.info is None and state.error is None:
                state.error = ValueError('請求已取消')
            with self.condition:
                if state.serial is not None:
                    self.by_serial.pop(state.serial, None)
                self.requests.pop(state.rid, None)
            state.prepared.set()
            state.done.set()

    def run(self):
        try:
            with self.engine.torch.inference_mode():
                self._run()
        except Exception:
            # An unexpected engine/iterate failure can leave the shared GPU state
            # inconsistent. Stop admission rather than reuse it for another request.
            error = traceback.format_exc()
            self.engine.error, self.engine.ready = error, False
            log('[EXL3] 生成服務錯誤：' + error)
            with self.condition:
                self.stopping = True
                remaining = list(self.requests.values())
            for state in remaining:
                try:
                    self.finish(state, RuntimeError(error), cancel=True)
                except Exception:
                    log(traceback.format_exc())

    def _run(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.requests or self.stopping)
                states = list(self.requests.values())
                if self.stopping and not states:
                    return
            for state in states:
                if state.cancelled.is_set():
                    self.finish(state, cancel=True)
                    continue
                if state.prepared.is_set():
                    continue
                try:
                    state.info = self.engine.prepare(state.body)
                    state.job = state.info[0]
                    state.serial = self.engine.generator.enqueue(state.job)
                    self.by_serial[state.serial] = state
                    state.body = None
                    state.prepared.set()
                    log(f'[EXL3] 請求 {state.rid}｜槽位 {state.slot + 1}/{self.parallel}')
                except (ValueError, TypeError, KeyError, AssertionError) as exc:
                    self.finish(state, ValueError(str(exc)), cancel=True)
            if not self.by_serial:
                continue
            discarded = set()
            gen, auto = self.engine.generator, self.engine.auto_draft
            if auto is not None:
                jobs = list(gen.active_jobs)
                ndt = gen.num_draft_tokens = auto.pick(len(jobs)) if jobs else auto.ceiling
                before = sum(j.new_tokens for j in jobs)
                acc0 = [getattr(j, 'accepted_draft_tokens', 0) for j in jobs]
                t0 = time.perf_counter()
            results = gen.iterate()
            if auto is not None and jobs:
                if not any(r.get('stage') == 'prefill' for r in results) and all(j in gen.active_jobs for j in jobs):
                    auto.record(len(jobs), ndt, sum(j.new_tokens for j in jobs) - before, time.perf_counter() - t0,
                                [getattr(j, 'accepted_draft_tokens', 0) - a for j, a in zip(jobs, acc0)])
            for result in results:
                if result.get('serial') in discarded:
                    continue
                state = self.by_serial.get(result.get('serial'))
                if state is None:
                    raise RuntimeError('生成結果缺少有效的請求 serial')
                if result.get('error') is not None:
                    discarded.add(state.serial)
                    self.finish(state, RuntimeError(str(result['error'])), cancel=True)
                    continue
                try:
                    state.results.put_nowait(result)
                except queue.Full:
                    state.cancelled.set()
                    discarded.add(state.serial)
                    self.finish(state, RuntimeError('客戶端讀取過慢，已停止此請求'), cancel=True)
                    continue
                if result.get('eos'):
                    self.finish(state)


class Engine:
    def __init__(self, args):
        self.args = args
        self.ready = False
        self.error = None
        self.parallel = getattr(args, 'parallel', 2)
        self.scheduler = None
        self.auto_draft = None
        self.latest = None
        self.vision = None                  # ExLlamaV3 視覺塔（--vision gpu／ram 才載入）
        self.images = OrderedDict()         # 圖片 sha256 → MMEmbedding：同一張圖在後續輪次沿用，前文快取才接得上

    def load_vision(self):
        """--vision gpu：視覺塔權重放顯卡；ram：權重留在系統 RAM（pinned），算圖時才串流上顯卡，幾乎不佔 VRAM。"""
        mode = self.args.vision
        if mode == 'off':
            return
        if getattr(self.config, 'vision', None) is None:
            log('[EXL3] 視覺：這個模型沒有視覺塔，只能聊文字')
            return
        from exllamav3 import Model
        torch = self.torch
        self.config.infer_params.vision_pinned = mode == 'ram'
        pp = self.config.vision_pp
        if self.args.vision_max_pixels:
            pp.max_pixels = max(pp.min_pixels, min(pp.max_pixels, self.args.vision_max_pixels))
        before = torch.cuda.memory_allocated()
        vision = Model.from_config(self.config, component='vision')
        vision.load(device='cuda:0')
        used = (torch.cuda.memory_allocated() - before) / 2 ** 30
        self.vision = vision
        where = '系統 RAM（算圖時串流上顯卡）' if mode == 'ram' else '顯卡'
        log(f'[EXL3] 視覺：已載入，權重放{where}，常駐顯存 {used:.2f} GiB；單張圖最多 {pp.max_pixels:,} 像素')

    def image_embedding(self, url):
        raw, img = decode_image(url)
        key = hashlib.sha256(raw).hexdigest()
        emb = self.images.get(key)
        if emb is None:
            t0 = time.perf_counter()
            emb = self.vision.get_image_embeddings(tokenizer=self.tokenizer, image=img)
            log(f'[EXL3] 圖片 {img.size[0]}×{img.size[1]} → {emb.embeddings.shape[0]} tok，'
                f'{time.perf_counter() - t0:.2f} s')
            self.images[key] = emb
            while len(self.images) > 16:
                self.images.popitem(last=False)
        else:
            self.images.move_to_end(key)
        return emb

    def load(self):
        try:
            import torch
            from exllamav3 import Generator, Job, model_init
            from exllamav3.constants import PAGE_SIZE
            self.torch, self.Job, self.page_size = torch, Job, PAGE_SIZE
            with torch.inference_mode():
                draft_path = self.args.draft_model_dir
                portable = None
                if draft_path:
                    dc = json.loads((Path(draft_path) / 'config.json').read_text(encoding='utf-8'))
                    if dc.get('architectures') == ['LiLiCorrDraftModel']:
                        from lilicorr import LiLiCorrEXL3, load_weights, reference_module
                        portable = LiLiCorrEXL3(draft_path, load_weights(Path(draft_path)), reference_module())
                        if self.args.chunk_size > portable.c.window:
                            self.args.chunk_size = portable.c.window
                        self.args.draft_model_dir = None
                        self.args.num_draft_tokens = min(max(1, self.args.num_draft_tokens or portable.c.slots), portable.c.slots)
                try:
                    load_args = copy.copy(self.args)
                    load_args.cache_size = self.args.cache_size * self.parallel
                    load_args.autosplit_max_batch_size = self.parallel
                    model, self.config, self.cache, self.tokenizer, draft_model, _, draft_cache = model_init.init(
                        load_args, min_draft_len=portable.c.slots if portable else None,
                        override_dynamic_seq_len=self.args.cache_size)
                finally:
                    self.args.draft_model_dir = draft_path
                if portable:
                    from types import SimpleNamespace
                    portable.attach_to(model)
                    draft_model = portable
                    draft_cache = SimpleNamespace(max_num_tokens=self.cache.max_num_tokens,
                        initialized=True, max_history=portable.c.slots)
                    log(f'[EXL3] LiLiCorr：原生 Windows，{portable.c.num_hidden_layers} 層，草稿 {self.args.num_draft_tokens} token')
                self.generator = Generator(model=model, cache=self.cache, tokenizer=self.tokenizer,
                    draft_model=draft_model, draft_cache=draft_cache, num_draft_tokens=self.args.num_draft_tokens,
                    max_batch_size=self.parallel, max_chunk_size=self.args.chunk_size,
                    cpu_cache_size=int((self.args.cpu_cache_size or 0) * 1024**3),
                    disk_cache_dir=self.args.disk_cache_dir if self.args.cpu_cache_size else None,
                    disk_cache_size=int((self.args.disk_cache_size or 50) * 1024**3),
                    recurrent_cache_size=int(max(0.0, self.args.recurrent_cache_size) * 1024 ** 3),
                    dynamic_draft_tokens=self.args.dynamic_draft, draft_confidence=self.args.draft_confidence,
                    ngram_match_min=self.args.ngram_match_min, hybrid_draft_tokens=self.args.hybrid_draft_tokens)
                log(f"[EXL3] 加速：多列快路徑 {'開' if os.environ.get('EXL3_GEMV_WMMA') != '0' else '關'}"
                    f"（上限 {os.environ.get('EXL3_GEMV_MAX_M')} 列，TN {os.environ.get('EXL3_GEMV_WMMA_TN', '1')}）"
                    f"｜#490 草稿查表放 RAM {'開' if os.environ.get('EXL3_DFLASH2_HOST_CB') == '1' else '關'}"
                    f"｜#517 分段搬運 {'開' if os.environ.get('EXL3_QC_STAGING_SLICED') == '1' else '關'}"
                    f"｜前文快取 RAM {self.args.cpu_cache_size or 0:g} GB"
                    + (f"／SSD {self.args.disk_cache_dir}（上限 {self.args.disk_cache_size or 50:g} GB）" if self.args.cpu_cache_size and self.args.disk_cache_dir else '')
                    + (f"｜重複內容長草稿 {self.args.hybrid_draft_tokens}" if self.args.hybrid_draft_tokens else '')
                    + (f"｜n-gram 草稿 {self.args.ngram_match_min}" if self.args.ngram_match_min else ''))
                if (self.args.auto_draft or self.args.auto_draft_accept) and self.generator.num_draft_tokens >= 1 and self.generator.draft_model is not None:
                    kind = AcceptDraft if self.args.auto_draft_accept else AutoDraft
                    self.auto_draft = kind(self.generator, self.generator.num_draft_tokens)
                    log(f'[EXL3] 自動草稿：開（' + ('依各位置猜中率' if self.args.auto_draft_accept else '依實測速度') + f'），檔位 {self.auto_draft.cands}')
                if self.generator.max_batch_size < self.parallel:
                    raise RuntimeError('引擎狀態槽數不足，無法提供指定的同時生成數')
                self.load_vision()
            mode = self.args.chat_mode
            self.hf_template, self.formatter = None, None
            source = load_hf_template(self.args.model_dir) if mode in ('auto', 'jinja') else None
            if source:
                self.hf_template = HFTemplate(source, self.tokenizer.bos_token, self.tokenizer.eos_token)
                mode = 'jinja'
            elif mode == 'jinja':
                raise ValueError('模型資料夾沒有 chat_template；請改選其他聊天模板')
            else:
                from chat_templates import prompt_formats
                if mode == 'auto':
                    cfg = json.loads((Path(self.args.model_dir) / 'config.json').read_text(encoding='utf-8'))
                    kind = cfg.get('model_type', '')
                    modes = {'qwen3_5': 'qwen35', 'qwen3': 'chatml', 'qwen2': 'chatml',
                        'llama': 'llama3', 'gemma': 'gemma', 'gemma2': 'gemma', 'gemma3': 'gemma',
                        'gemma4': 'gemma4', 'gpt_oss': 'gptoss', 'glm4': 'glm', 'phi3': 'phi'}
                    if kind not in modes:
                        raise ValueError(f'模型沒有 chat_template，也沒有 {kind} 的內建模板；請手動選擇聊天模板')
                    mode = modes[kind]
                if mode not in prompt_formats:
                    raise ValueError(f'未知聊天模板：{mode}')
                self.formatter = prompt_formats[mode]('User', 'Assistant')
            self.mode = mode
            stops = list(self.config.eos_token_id_list or [])
            if self.tokenizer.eos_token_id is not None:
                stops.append(self.tokenizer.eos_token_id)
            self.stop_ids = sorted(set(stops))
            self.draft_tokens = self.generator.max_draft_tokens
            self.scheduler = GenerationScheduler(self, self.parallel)
            self.ready = True
            log(f'[EXL3] 同時生成 {self.parallel} 路｜每路 Context {self.args.cache_size}｜'
                f'總 KV 容量 {self.cache.max_num_tokens} token；主模型與草稿各載一份')
            audit = globals().get('runtime_write_audit')
            if audit:
                audit()
            print('EXL3_SERVER_READY', json.dumps(self.props(), ensure_ascii=False), flush=True)
        except Exception:
            self.ready = False
            if self.scheduler:
                self.scheduler.close()
            self.error = traceback.format_exc()
            print('EXL3_SERVER_LOAD_ERROR', self.error, flush=True)

    def slots(self):
        active = {s['id']: s for s in self.scheduler.snapshot()} if self.scheduler else {}
        return [dict(id=i, n_ctx=self.args.cache_size, is_processing=False, request_id=None)
                | active.get(i, {}) for i in range(self.parallel)]

    def props(self):
        active = [s for s in self.slots() if s['is_processing']]
        request_ids = [s['request_id'] for s in active]
        return {'backend': 'exllamav3', 'status': 'ok' if self.ready else ('error' if self.error else 'loading'),
            'error': self.error, 'model_path': self.args.model_dir, 'alias': self.args.alias,
            'default_generation_settings': {'n_ctx': self.args.cache_size, 'params': {
                'temperature': self.args.temperature, 'top_p': self.args.top_p, 'top_k': self.args.top_k,
                'min_p': self.args.min_p, 'n_predict': -1}},
            'total_slots': self.parallel, 'active_slots': len(active),
            'total_context': self.args.cache_size * self.parallel, 'model_alias': self.args.alias,
            'modalities': {'vision': self.vision is not None, 'audio': False},
            'vision': self.args.vision if self.vision is not None else 'off',
            'build_info': 'exllamav3',
            'draft_model': self.args.draft_model_dir, 'chat_mode': getattr(self, 'mode', self.args.chat_mode),
            'thinking': self.args.thinking != 'off', 'busy': bool(active),
            'request_id': request_ids[0] if len(request_ids) == 1 else None,
            'request_ids': request_ids, 'latest': self.latest}

    def prepare(self, body):
        if body.get('response_format') not in (None, {'type': 'text'}):
            raise ValueError('EXL3 尚未支援 response_format（JSON 模式）')
        tools = request_tools(body.get('tools'), body.get('tool_choice'))
        if tools and not self.hf_template:
            raise ValueError('工具呼叫需要模型自帶的聊天模板（chat_template.jinja）')
        if body.get('n', 1) != 1:
            raise ValueError('n 必須是 1')
        kwargs = body.get('chat_template_kwargs') or {}
        if not isinstance(kwargs, dict) or not all(isinstance(k, str) for k in kwargs):
            raise ValueError('chat_template_kwargs 必須是 JSON object')
        think, effort, requested_effort = resolve_reasoning(
            body, kwargs, self.args.reasoning_effort, self.args.thinking != 'off',
            getattr(self.hf_template, 'reasoning_efforts', ()))
        embeddings = []
        image = None
        if self.vision is not None:
            def image(url):
                emb = self.image_embedding(url)
                if all(e is not emb for e in embeddings):
                    embeddings.append(emb)
                return emb.text_alias
        if self.hf_template:
            extra = {k: v for k, v in kwargs.items() if k not in
                     ('messages', 'enable_thinking', 'add_generation_prompt', 'reasoning_effort')}
            if effort is not None:
                extra['reasoning_effort'] = effort
            try:
                text = self.hf_template.render(template_messages(body.get('messages'), image), think, extra, tools)
            except ValueError:
                raise
            except Exception as exc:
                raise ValueError(f'聊天模板：{exc}') from exc
            implicit = text.rstrip().endswith('<think>')
            ids = self.tokenizer.encode(text, add_bos=False, encode_special_tokens=True, embeddings=embeddings or None)
        else:
            system, pairs = prompt_messages(text_messages(body.get('messages')))
            text = self.formatter.format(system or self.formatter.default_system_prompt(think), pairs, think)
            implicit = bool(think and self.formatter.thinktag()[0] and self.formatter.inject_thinktag())
            if implicit:
                text += self.formatter.thinktag()[0]
            ids = self.tokenizer.encode(text, add_bos=self.formatter.add_bos(), encode_special_tokens=True)
        requested = body.get('max_completion_tokens', body.get('max_tokens', body.get('n_predict', 0)))
        context = min(getattr(self.args, 'cache_size', self.generator.max_total_tokens), self.generator.max_total_tokens)
        limit = response_budget(ids.shape[-1], context, requested, self.draft_tokens)
        sampler_args = copy.copy(self.args)
        mapping = {'temperature': 'temperature', 'top_p': 'top_p', 'top_k': 'top_k', 'min_p': 'min_p',
                   'presence_penalty': 'presence_penalty', 'frequency_penalty': 'frequency_penalty',
                   'repetition_penalty': 'repetition_penalty'}
        for key, attr in mapping.items():
            if key in body:
                val = body[key]
                if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
                    raise ValueError(f'{key} 必須是有限數值')
                setattr(sampler_args, attr, val)
        if sampler_args.temperature < 0 or not 0 < sampler_args.top_p <= 1 or not 0 <= sampler_args.min_p <= 1:
            raise ValueError('temperature >= 0，0 < top_p <= 1，0 <= min_p <= 1')
        if int(sampler_args.top_k) != sampler_args.top_k or sampler_args.top_k < 0:
            raise ValueError('top_k 必須是非負整數')
        sampler_args.top_k = int(sampler_args.top_k)
        from exllamav3 import GreedySampler, model_init
        sampler = GreedySampler() if sampler_args.temperature == 0 else model_init.get_arg_sampler(sampler_args)
        stop = list(self.stop_ids)
        if self.formatter:
            stop += [x for x in self.formatter.stop_conditions(self.tokenizer) if x and x not in stop]
        extra_stop = body.get('stop', [])
        if isinstance(extra_stop, str):
            extra_stop = [extra_stop]
        if not isinstance(extra_stop, list) or not all(isinstance(s, str) and s for s in extra_stop):
            raise ValueError('stop 必須是非空字串或字串陣列')
        seed = body.get('seed')
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise ValueError('seed 必須是整數')
        actual_effort = (effort or '預設') if think else '關閉思考'
        if think and not self.hf_template:
            actual_effort = '內建模板（不區分強度）'
        log(f"[思考] 請求 {requested_effort or 'default'} → 模型 {actual_effort}")
        return self.Job(input_ids=ids, max_new_tokens=limit, sampler=sampler,
            stop_conditions=stop + extra_stop, seed=seed, embeddings=embeddings or None), limit, implicit, \
            int(ids.shape[-1]), tools


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    QUIET_GETS = {'/props', '/health', '/slots', '/v1/models', '/models'}

    def log_message(self, fmt, *args):
        log(f'HTTP {self.address_string()} {fmt % args}')

    def log_request(self, code='-', size='-'):
        # GGUFRun 介面每 2 秒查 /props、網頁也會輪詢：成功的狀態查詢不記，其餘照記
        try:
            ok = 200 <= int(code) < 400
        except (TypeError, ValueError):
            ok = False
        if ok and self.command == 'GET' and (self.route() in self.QUIET_GETS or self.route() in UI_FILES):
            return
        super().log_request(code, size)

    def route(self):
        return urlsplit(self.path).path.rstrip('/') or '/'

    def send_ui(self, path):
        meta = UI_FILES.get('/' if path == '/index.html' else path)
        if not meta:
            return False
        data = (UI_DIR / meta['file']).read_bytes()
        gz = meta.get('gzip', False)
        if gz and 'gzip' not in self.headers.get('Accept-Encoding', ''):
            import gzip
            data, gz = gzip.decompress(data), False
        self.send_response(200)
        self.send_header('Content-Type', meta.get('type') or 'application/octet-stream')
        if gz:
            self.send_header('Content-Encoding', 'gzip')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'public, max-age=31536000, immutable' if '/immutable/' in path else 'no-cache')
        self.end_headers()
        self.wfile.write(data)
        return True

    def send_json(self, status, data):
        payload = json.dumps(data, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(payload)))
        if self.close_connection:
            self.send_header('Connection', 'close')
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(payload)

    def read_body(self):
        n = int(self.headers.get('Content-Length', '0'))
        cap = 64 if getattr(self.server.engine, 'vision', None) is not None else 8   # 開視覺時圖片 base64 放在 JSON 裡
        if not 0 < n <= cap * 1024 * 1024:
            raise ValueError(f'JSON 請求大小須介於 1 byte 與 {cap} MiB')
        body = json.loads(self.rfile.read(n))
        if not isinstance(body, dict):
            raise ValueError('請求必須是 JSON object')
        origin = self.headers.get('Origin')
        if origin and origin not in {f'http://127.0.0.1:{self.server.server_port}', f'http://localhost:{self.server.server_port}'}:
            raise ValueError('請從本服務網頁送出請求')
        return body

    def do_GET(self):
        engine = self.server.engine
        path = self.route()
        if path in ('/exl3', '/chat.html') or (path == '/' and not UI_FILES):
            page = (ROOT / 'web/chat.html').read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(page)))
            self.end_headers()
            self.wfile.write(page)
        elif self.send_ui(path):
            pass
        elif path == '/health':
            self.send_json(200 if engine.ready else 503, engine.props())
        elif path == '/props':
            if not engine.ready and self.headers.get('Sec-Fetch-Mode'):
                # 瀏覽器（llama.cpp 網頁）比照 llama-server：載入中回 503，網頁會顯示載入中並自動重試；
                # GGUFRun 介面（Python urllib，無 Sec-Fetch-*）照舊拿 200＋status 顯示進度
                self.send_json(503, {'error': {'code': 503, 'type': 'unavailable_error',
                                               'message': engine.error or 'Loading model'}})
            else:
                self.send_json(200, engine.props())
        elif path in ('/v1/models', '/models'):
            self.send_json(200, {'object': 'list', 'data': [{'id': engine.args.alias, 'object': 'model', 'owned_by': 'local',
                'context_length': engine.args.cache_size}]})
        elif path == '/slots':
            # llama.cpp 網頁用來判斷是否閒置
            self.send_json(200, engine.slots())
        elif path == '/tools':
            # 403＝伺服器端工具未開啟；llama.cpp 網頁收到後安靜略過
            self.send_json(403, {'error': {'message': '此伺服器未提供伺服器端工具'}})
        else:
            self.send_json(404, {'error': {'message': 'Unknown endpoint'}})

    def disconnected(self):
        try:
            return bool(select.select([self.connection], [], [], 0)[0]
                        and self.connection.recv(1, socket.MSG_PEEK) == b'')
        except OSError:
            return True

    def do_POST(self):
        engine = self.server.engine
        if self.route() == '/tools':
            self.close_connection = True
            self.send_json(403, {'error': {'message': '此伺服器未提供伺服器端工具'}})
            return
        try:
            body = self.read_body()
        except (ValueError, TypeError, UnicodeError) as exc:
            self.close_connection = True
            self.send_json(400, {'error': {'message': str(exc)}})
            return
        path = self.route()
        if path == '/v1/streams/lookup':
            self.send_json(200, [])                     # 不保存可續傳的串流
            return
        if path == '/v1/chat/completions/control':
            self.send_json(200, {'success': False, 'error': {'message': 'EXL3 不支援中途結束思考'}})
            return
        if path == '/cancel':
            requested = body.get('request_id')
            ok = bool(isinstance(requested, str) and engine.scheduler and engine.scheduler.cancel(requested))
            self.send_json(200, {'cancelled': ok})
            return
        if path != '/v1/chat/completions':
            self.send_json(404, {'error': {'message': 'Unknown endpoint'}})
            return
        if not engine.ready:
            self.send_json(503, {'error': {'code': 503, 'type': 'unavailable_error',
                                           'message': engine.error or '模型載入中，請稍候再送出'}})
            return
        if body.get('n_predict') == 0 and not body.get('stream'):
            # llama.cpp 網頁打字時的預先編碼請求：直接回空結果，不佔用生成槽位
            self.send_json(200, {'id': 'chatcmpl-' + uuid.uuid4().hex, 'object': 'chat.completion',
                'created': int(time.time()), 'model': engine.args.alias,
                'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': ''}, 'finish_reason': 'length'}],
                'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}})
            return
        try:
            state = engine.scheduler.submit(body)
        except RuntimeError as exc:
            self.send_json(503, {'error': {'message': str(exc)}})
            return
        if state is None:
            self.send_json(409, {'error': {'message': '同時生成槽位已滿；請稍後重試或停止其中一個請求'}})
            return
        job, streaming, started = None, False, time.perf_counter()
        try:
            while not state.prepared.wait(timeout=0.1):
                if self.disconnected():
                    state.cancelled.set()
                    return
            if state.error is not None:
                raise state.error
            job, capacity, implicit, prompt_count, tools = state.info
            rid = state.rid
            created = int(time.time())
            live = LiveTimings(job, prompt_count, getattr(engine, 'page_size', 256))
            progress_log = ProgressLog(live, request_id=rid)
            want_progress = bool(body.get('return_progress'))
            clock = {'write': time.perf_counter(), 'timings': 0.0}
            def event(delta, finish=None, usage=None, metrics=None, timings=None, progress=None):
                obj = {'id': rid, 'object': 'chat.completion.chunk', 'created': created, 'model': engine.args.alias,
                    'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish}]}
                if usage is not None:
                    obj['usage'] = usage
                if timings is not None:
                    obj['timings'] = timings
                if progress is not None:
                    obj['prompt_progress'] = progress
                if metrics is not None:
                    obj['exl3'] = metrics
                self.wfile.write(('data: ' + json.dumps(obj, ensure_ascii=False) + '\n\n').encode('utf-8'))
                self.wfile.flush()
                clock['write'] = time.perf_counter()
            if body.get('stream', False):
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream; charset=utf-8')
                self.send_header('Cache-Control', 'no-cache')
                self.send_header('Connection', 'close')
                self.end_headers()
                self.close_connection = True
                streaming = True
                event({'role': 'assistant'}, metrics={'response_capacity': capacity})
            parsed, output, terminal = ReasoningParser(implicit), {'content': '', 'reasoning_content': ''}, None
            calls_parser, calls = (ToolCallParser(tools) if tools else None), []
            def emit(pieces, final=False):
                for key, chunk in pieces:
                    if key == 'content' and calls_parser:
                        sub = calls_parser.feed(chunk)
                    else:
                        sub = [(key, chunk)]
                    for k, c in sub:
                        send(k, c)
                if final and calls_parser:
                    for k, c in calls_parser.feed('', final=True):
                        send(k, c)
            def send(key, chunk):
                if key == 'tool_call':
                    calls.append(chunk)
                    if streaming:
                        event({'tool_calls': [chunk]})
                    return
                output[key] += chunk
                if streaming:
                    now = time.perf_counter()
                    timings = None
                    if now - clock['timings'] >= 0.2:            # 網頁即時 tok/s；約每秒 5 次
                        clock['timings'] = now
                        timings = live.timings()
                    event({key: chunk}, timings=timings)
            while not state.done.is_set() or not state.results.empty():
                if self.disconnected():
                    state.cancelled.set()
                    log(speed_line(live.timings(), True, '網頁中斷連線'))
                    return
                if streaming and time.perf_counter() - clock['write'] >= 1.0:
                    self.wfile.write(b': ping\n\n')              # SSE 註解：長預填期間保持連線
                    self.wfile.flush()
                    clock['write'] = time.perf_counter()
                try:
                    result = state.results.get(timeout=0.1)
                except queue.Empty:
                    continue
                stage = result.get('stage')
                if stage == 'prefill':
                    progress_log.prefill(result)
                    if streaming and want_progress:
                        event({}, progress=live.progress(result))
                if stage == 'streaming':
                    emit(parsed.feed(result.get('text', '')))
                    progress_log.generate()
                if result.get('eos'):
                    terminal = result
            if state.error is not None:
                raise state.error
            emit(parsed.feed('', final=True), final=True)
            terminal = terminal or {}
            usage = {'prompt_tokens': terminal.get('prompt_tokens', prompt_count),
                'completion_tokens': terminal.get('new_tokens', live.timings()['predicted_n'])}
            usage['total_tokens'] = usage['prompt_tokens'] + usage['completion_tokens']
            seconds = terminal.get('time_generate', 0)
            reason = 'stop' if terminal.get('eos_reason') != 'max_new_tokens' else 'length'
            if not terminal:
                reason = 'stop'
            if calls and reason == 'stop':
                reason = 'tool_calls'
            metrics = {'response_capacity': capacity, 'elapsed_seconds': time.perf_counter() - started,
                'generate_tps': usage['completion_tokens'] / seconds if seconds else None,
                'prefill_seconds': terminal.get('time_prefill'),
                'prefill_tps': (usage['prompt_tokens'] - (terminal.get('cached_tokens') or 0)) / terminal['time_prefill']
                    if terminal.get('time_prefill') else None,
                'cached_tokens': terminal.get('cached_tokens'),
                'accepted_draft_tokens': terminal.get('accepted_draft_tokens'),
                'rejected_draft_tokens': terminal.get('rejected_draft_tokens'), 'cancelled': state.cancelled.is_set()}
            engine.latest = {'request_id': rid, 'usage': usage, **metrics}
            timings = live.timings(terminal or None)
            if streaming:
                event({}, reason, usage=usage, metrics=metrics, timings=timings)
                self.wfile.write(b'data: [DONE]\n\n')
                self.wfile.flush()
            else:
                self.send_json(200, {'id': rid, 'object': 'chat.completion', 'created': created, 'model': engine.args.alias,
                    'choices': [{'index': 0, 'message': {'role': 'assistant', **output,
                        **({'tool_calls': [{k: v for k, v in c.items() if k != 'index'} for c in calls]} if calls else {})},
                        'finish_reason': reason}],
                    'usage': usage, 'timings': timings, 'exl3': metrics})
            log(f'[{rid}] ' + speed_line(timings, state.cancelled.is_set(), '按下停止'))
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        except (ValueError, TypeError, KeyError) as exc:
            if streaming:
                self.wfile.write(('data: ' + json.dumps({'error': {'message': str(exc)}}) + '\n\n').encode('utf-8'))
            else:
                self.send_json(400, {'error': {'message': str(exc)}})
        except Exception as exc:
            traceback.print_exc()
            if streaming:
                self.wfile.write(('data: ' + json.dumps({'error': {'message': str(exc)}}) + '\n\n').encode('utf-8'))
                self.wfile.flush()
            else:
                self.send_json(500, {'error': {'message': str(exc)}})
        finally:
            if not state.done.is_set():
                state.cancelled.set()


def main():
    try:
        sys.stdout.reconfigure(newline='\n')            # 管線輸出用 \n，GGUFRun 介面不會多出空行
    except (AttributeError, ValueError):
        pass
    from exllamav3 import model_init
    parser = argparse.ArgumentParser(allow_abbrev=False)
    model_init.add_args(parser, cache=True, add_sampling_args=True, add_draft_model_args=True,
        default_cache_size=32768, default_recurrent_cache_size=4.0, default_chunk_size=512)
    parser.add_argument('--port', type=int, default=18437)
    parser.add_argument('--host', choices=['127.0.0.1'], default='127.0.0.1')
    parser.add_argument('--alias', default=None)
    parser.add_argument('--chat-mode', default='auto')
    parser.add_argument('--thinking', choices=['on', 'off', 'auto'], default='on')
    parser.add_argument('--reasoning-effort', default=None)
    parser.add_argument('--parallel', type=int, choices=range(1, 9), default=2,
                        help='同時生成數（1～8）；-cs 是每路 Context，總 KV 容量為 -cs × 路數')
    parser.add_argument('--auto-draft', action='store_true',
                        help='依實測速度自動調草稿數（0～-ndt，0＝暫停草稿；依同時生成路數分開量）；預設關')
    parser.add_argument('--auto-draft-accept', action='store_true',
                        help='依最近各位置的草稿猜中率與列數級距耗時，每輪自動調草稿數（0～-ndt）；換內容幾輪內跟上；預設關')
    parser.add_argument('--gemv-wmma', choices=['0', '1'], default=None,
                        help='多列快路徑（8～64 列 WMMA GEMV）；預設 1＝開')
    parser.add_argument('--gemv-max-m', choices=['32', '64'], default=None,
                        help='多列快路徑一次最多幾列；預設 64')
    parser.add_argument('--gemv-wmma-tn', choices=['1', '2', '4'], default=None,
                        help='多列快路徑每個 GPU 工作組一次算幾塊輸出（輸入共用）；1＝原做法（預設），2／4 只用於 1～4 bit 權重')
    parser.add_argument('--dflash2-host-cb', action='store_true',
                        help='實驗性（#490）：DFlash2 草稿的選字查表留在系統 RAM，省約 240 MB 顯存')
    parser.add_argument('--qc-staging-sliced', action='store_true',
                        help='實驗性（#517）：量化 KV 在注意力計算時分固定大小的段搬運')
    parser.add_argument('--vision', choices=['off', 'gpu', 'ram'], default='off',
                        help='視覺塔：off 不載入；gpu 權重放顯卡；ram 權重放系統 RAM，算圖時串流上顯卡')
    parser.add_argument('--vision-max-pixels', type=int, default=2097152,
                        help='單張圖縮到這個像素數以內（約每 1024 像素 1 個 token）；0＝用模型預設')
    args = parser.parse_args()
    args.alias = args.alias or Path(args.model_dir).name
    if args.cache_size < 256 or args.cache_size % 256:
        parser.error('每路 Context（-cs）必須至少 256，且為 256 的倍數')
    if not 1 <= args.port <= 65535:
        parser.error('port 必須介於 1 與 65535')
    engine = Engine(args)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    server.engine = engine
    print('EXL3_HTTP_LISTEN', args.host, args.port, flush=True)
    threading.Thread(target=engine.load, daemon=True).start()
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        if engine.scheduler:
            engine.scheduler.close()
        server.server_close()


if __name__ == '__main__':
    main()
