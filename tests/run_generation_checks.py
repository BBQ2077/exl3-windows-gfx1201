"""Bounded native-Windows model checks using ExLlamaV3's public loader/generator APIs."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app/examples'))
from chat_templates import PromptFormat_qwen35
import torch
from exllamav3 import Generator, Job, GreedySampler, model_init


def now():
    return datetime.now().astimezone().isoformat()


def strip_fence(text):
    match = re.fullmatch(r'\s*```(?:python|json)?\s*\n(.*?)\n```\s*', text, re.S)
    return match.group(1) if match else text.strip()


def validate_add(text):
    source = strip_fence(text)
    tree = ast.parse(source)
    allowed = (ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return,
               ast.BinOp, ast.Add, ast.Name, ast.Load, ast.Constant, ast.Expr)
    if not all(isinstance(node, allowed) for node in ast.walk(tree)):
        raise AssertionError('Function contains syntax outside the arithmetic test allowlist')
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise AssertionError('Expected one function')
    namespace = {'__builtins__': {}, 'int': int, 'float': float}
    exec(compile(tree, '<generated-add>', 'exec'), namespace)
    function = namespace['add']
    for a, b in [(2, 3), (-4, 7), (0, 0)]:
        if function(a, b) != a + b:
            raise AssertionError('Arithmetic function result mismatch')


@torch.inference_mode()
def run(args, report, save):
    started = time.perf_counter()
    report['vram_before_load'] = dict(zip(['free_bytes', 'total_bytes'], torch.cuda.mem_get_info()))
    model, config, cache, tokenizer, draft_model, draft_config, draft_cache = model_init.init(args)
    generator = Generator(
        model=model, cache=cache, tokenizer=tokenizer,
        draft_model=draft_model, draft_cache=draft_cache,
        num_draft_tokens=args.num_draft_tokens,
        max_batch_size=1, max_chunk_size=args.chunk_size,
        recurrent_cache_size=0, cpu_cache_size=0,
    )
    report['load_seconds'] = time.perf_counter() - started
    report['draft_active'] = bool(generator.dflash_draft)
    report['gpu'] = torch.cuda.get_device_name(0)
    report['torch'] = torch.__version__
    report['hip'] = torch.version.hip
    report['cache_tokens'] = args.cache_size
    report['cache_quant'] = args.cache_quant
    report['graphs'] = os.environ.get('EXL3_GRAPHS')
    print('MODEL_LOADED', json.dumps({k: report[k] for k in ['load_seconds', 'draft_active', 'gpu']}, ensure_ascii=False), flush=True)
    save()
    formatter = PromptFormat_qwen35('User', 'Assistant')
    stop = config.eos_token_id_list

    def generate(name, question, limit, measured=True):
        prompt = formatter.format('You are a helpful AI assistant.', [(question, None)], False)
        ids = tokenizer.encode(prompt, add_bos=False, encode_special_tokens=True)
        job = Job(input_ids=ids, max_new_tokens=limit, stop_conditions=stop,
                  sampler=GreedySampler(), seed=1234)
        generator.enqueue(job)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        first_chunk_time = None
        first_chunk_tokens = 0
        chunks = []
        terminal = None
        last_progress = t0
        print('CASE_BEGIN', name, 'prompt_tokens', ids.shape[-1], 'max_new_tokens', limit, flush=True)
        while generator.num_remaining_jobs():
            for result in generator.iterate():
                if result.get('stage') == 'streaming':
                    chunks.append(result.get('text', ''))
                    token_ids = result.get('token_ids')
                    if first_chunk_time is None and token_ids is not None and token_ids.numel():
                        first_chunk_time = time.perf_counter()
                        first_chunk_tokens = token_ids.numel()
                if result.get('eos'):
                    terminal = result
            current = time.perf_counter()
            if current - last_progress >= 10:
                print('CASE_PROGRESS', name, 'seconds', round(current - t0, 1), flush=True)
                last_progress = current
            if current - t0 > 240:
                generator.cancel(job)
                raise TimeoutError(f'Case exceeded 240 seconds: {name}')
        torch.cuda.synchronize()
        end = time.perf_counter()
        if terminal is None:
            raise AssertionError('No completed generation event')
        text = ''.join(chunks).strip()
        n = terminal['new_tokens']
        record = {
            'name': name, 'prompt': question, 'max_new_tokens': limit,
            'prompt_tokens': terminal['prompt_tokens'],
            'cached_tokens': terminal['cached_tokens'], 'new_tokens': n,
            'time_prefill': terminal['time_prefill'],
            'time_generate': terminal['time_generate'],
            'generate_tps': n / terminal['time_generate'],
            'wall_seconds': end - t0,
            'first_chunk_ms': (first_chunk_time - t0) * 1000 if first_chunk_time is not None else None,
            'eos_reason': terminal.get('eos_reason'),
            'text': text, 'text_sha256': hashlib.sha256(text.encode('utf-8')).hexdigest(),
        }
        if first_chunk_time is not None and n > first_chunk_tokens:
            record['post_first_chunk_tps'] = (n - first_chunk_tokens) / (end - first_chunk_time)
        if 'accepted_draft_tokens' in terminal:
            accepted = terminal['accepted_draft_tokens']
            rejected = terminal['rejected_draft_tokens']
            record.update(accepted_draft_tokens=accepted, rejected_draft_tokens=rejected,
                          draft_acceptance=accepted / (accepted + rejected) if accepted + rejected else None)
        if not text or n <= 0:
            raise AssertionError('Empty generated response')
        if measured:
            report['cases'].append(record)
            save()
        print('CASE_END', name, 'tokens', n, 'generate_tps', round(record['generate_tps'], 2), 'acceptance', record.get('draft_acceptance'), flush=True)
        if name.startswith('check_'):
            print('RESPONSE', json.dumps(text, ensure_ascii=False), flush=True)
        return record

    generate('warmup', 'List the first ten positive integers, separated by commas.', 16, False)
    checks = [
        ('check_chinese', '請用繁體中文回答：台灣的首都是哪個城市？只回答城市名稱，不要解釋。', 32,
         lambda text: any(city in text for city in ['台北', '臺北', 'Taipei'])),
        ('check_json', 'Output exactly this JSON object and nothing else: {"status":"ok","value":42}', 64,
         lambda text: json.loads(strip_fence(text)) == {'status': 'ok', 'value': 17 + 25}),
        ('check_python', 'Write only a Python function named add with parameters a and b that returns a + b. No type hints, docstrings, comments, imports, tests or explanation.', 96,
         lambda text: validate_add(text) is None),
    ]
    for name, question, limit, validator in checks:
        record = generate(name, question, limit)
        record['validation_pass'] = bool(validator(record['text']))
        save()
        if not record['validation_pass']:
            raise AssertionError(f'Output validation failed: {name}')
    workloads = {
        'code': 'Write a Python class implementing an LRU cache with get, put and delete, using a dict and a doubly linked list, with type hints and docstrings. Then write pytest tests for it.',
        'prose': 'Explain how a home Wi-Fi router forwards packets between a laptop and a website. Write at least six detailed paragraphs for a technically curious beginner, without code.',
    }
    for workload, question in workloads.items():
        for repeat in range(args.repeats):
            generate(f'{workload}_{repeat + 1}', question, args.max_new_tokens)
    report['summary'] = {}
    for workload in workloads:
        cases = [c for c in report['cases'] if c['name'].startswith(workload + '_')]
        report['summary'][workload] = {
            'runs': len(cases), 'new_tokens': sum(c['new_tokens'] for c in cases),
            'generate_tps': sum(c['new_tokens'] for c in cases) / sum(c['time_generate'] for c in cases),
        }
    report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    report['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
    report['status'] = 'PASS'
    print('GENERATION_FLOW_PASS', json.dumps(report['summary']), flush=True)


def main():
    import msvcrt
    parser = argparse.ArgumentParser(allow_abbrev=False)
    model_init.add_args(parser, cache=True, add_draft_model_args=True, default_cache_size=2048,
                        default_recurrent_cache_size=0, default_chunk_size=512)
    parser.add_argument('--label', required=True)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--max-new-tokens', type=int, default=256)
    args = parser.parse_args()
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', args.label):
        raise ValueError('Invalid report label')
    target = ROOT / 'logs' / f'generation-{args.label}.json'
    report = {'started_at': now(), 'label': args.label, 'status': 'RUNNING', 'args': vars(args), 'cases': []}
    def save():
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    with (ROOT / 'logs/generation-test.lock').open('a+b') as lock:
        lock.seek(0)
        if not lock.read(1):
            lock.write(b'0')
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            save()
            run(args, report, save)
        except BaseException:
            report['status'] = 'FAIL'
            report['error'] = traceback.format_exc()
            raise
        finally:
            report['finished_at'] = now()
            save()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


if __name__ == '__main__':
    main()
