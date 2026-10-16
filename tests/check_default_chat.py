"""Exercise the default Qwen launcher using only the bundled Python."""
from pathlib import Path
import json
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
log = ROOT / 'logs/default-chat-test.log'
response = ROOT / 'logs/default-chat-response.txt'
background = ' '.join(f'資料項目{i}：本項目僅供背景閱讀，不影響最後的算術題。' for i in range(160))
prompt = background + ' 請解最後這題：小明有17顆球，送出5顆，再買9顆，現在有幾顆？請先思考，再用繁體中文簡短回答。'
cmd = ['cmd.exe', '/d', '/c', str(ROOT / 'chat-qwen38.cmd'), '-basic', '-temp', '0',
       '-prompt', prompt, '-save', str(response)]
started = time.perf_counter()
with log.open('wb') as stream:
    child = subprocess.Popen(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    print('DEFAULT_CHAT_PID', child.pid, flush=True)
    try:
        code = child.wait(timeout=480)
    except subprocess.TimeoutExpired:
        subprocess.run(['taskkill.exe', '/PID', str(child.pid), '/T', '/F'], capture_output=True)
        child.wait()
        raise
text = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', log.read_text(encoding='utf-8', errors='replace'))
saved = response.read_text(encoding='utf-8') if response.exists() else ''
budget = re.search(r'Context: ([\d,]+) tokens; prompt: ([\d,]+); response capacity: ([\d,]+)', text)
audit = json.loads((ROOT / 'logs/last-audit.json').read_text(encoding='utf-8'))
audit_path = ROOT / 'logs/default-chat-audit.json'
audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
assert json.loads(audit_path.read_text(encoding='utf-8')) == audit
values = [int(x.replace(',', '')) for x in budget.groups()] if budget else [0,0,0]
result = {'exit_code': code, 'elapsed_seconds': time.perf_counter() - started,
          'context_tokens': values[0], 'prompt_tokens': values[1], 'response_capacity': values[2],
          'thinking_present': '</think>' in saved and bool(saved.split('</think>')[0].strip()),
          'correct_answer': '21' in saved.split('</think>')[-1], 'draft_active': 'Draft:' in text,
          'isolation_pass': audit['isolation_pass'], 'output': saved, 'log': str(log),
          'audit_file': str(audit_path)}
result['pass'] = (code == 0 and values[0] == 32768 and values[1] > 2048 and values[2] > 16384
                  and result['thinking_present'] and result['correct_answer'] and result['draft_active']
                  and result['isolation_pass'] and 'Response reached its' not in text)
p = ROOT / 'logs/default-chat-check.json'
p.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
assert json.loads(p.read_text(encoding='utf-8')) == result
print(text[-4000:], flush=True)
print('DEFAULT_CHAT_CHECK', json.dumps(result, ensure_ascii=False), flush=True)
raise SystemExit(0 if result['pass'] else 1)
