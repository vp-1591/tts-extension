"""End-to-end test of the extension's auto-start / auto-revive flow via Playwright.

Runs against Playwright's bundled Chromium: branded Chrome >=137 ignores
--load-extension and hides automation-installed extensions from normal tab
navigation, while Chromium's classic --load-extension path exposes the
extension (service worker + chrome-extension:// pages) exactly like a manual
"Load unpacked" does.

Stages (each prints PASS/FAIL):
  1. preconditions: server offline; Chromium resolves host manifest via registry
  2. open panel.html as a tab with the server stopped -> panel reaches 'Server online'
  3. kill the server process -> panel flips offline (3-strike heartbeat) -> auto-revive

Run:  PYTHONDONTWRITEBYTECODE=1 python tests/e2e/playwright_e2e.py
"""

import sys
import time
import winreg
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'native_host'))
from tts_native_host import HOST_NAME

REPO = Path(__file__).resolve().parents[2]
EXT_ID = 'habcfbjmnckpiaockfecaibphndkacac'
HEALTH_URL = 'http://127.0.0.1:5912/health'
STATUS_SELECTOR = '#status'
ONLINE_TEXT = 'Server online'

failures = []
results = []


def check(name, ok, detail=''):
    results.append((name, ok, detail))
    print(('PASS' if ok else 'FAIL') + f'  {name}' + (f'  [{detail}]' if detail else ''))
    if not ok:
        failures.append(name)


def find_server_pid():
    import subprocess
    net = subprocess.run(['netstat', '-ano', '-p', 'tcp'], capture_output=True, text=True).stdout
    for line in net.splitlines():
        if '5912' in line and 'LISTENING' in line:
            return int(line.split()[-1])
    return None


def wait_for_status(page, texts, timeout_s=120):
    """Return the first matched status text, or the last observed one on timeout.
    Logs every status change with a wall-clock time to reconstruct the flow."""
    deadline = time.monotonic() + timeout_s
    last = ''
    while time.monotonic() < deadline:
        try:
            cur = page.locator(STATUS_SELECTOR).inner_text(timeout=2000)
        except Exception as e:
            cur = f'<page error: {e}>'
        if cur != last:
            print(f'    [{time.strftime("%H:%M:%S")}] status: {cur.strip()!r}')
            last = cur
        if any(t in last for t in texts):
            return last
        time.sleep(2)
    return last


def attach_console(page, name):
    page.on('console', lambda m: print(f'    [{name} console] {m.type}: {m.text}'))
    page.on('pageerror', lambda e: print(f'    [{name} pageerror] {e}'))


def stage_clean_copy():
    """Copy manifest-referenced files only — the repo root holds .bmad/ and .claude
    __pycache__ dirs, and Chrome refuses to load unpacked extensions containing any
    entry whose name starts with '_'."""
    import shutil
    dst = REPO / 'tmp' / 'pw-ext'
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    (dst / 'icons').mkdir()
    for rel in ['manifest.json', 'panel.html', 'panel.js', 'background.js',
                'icons/icon128.png']:
        shutil.copy2(REPO / rel, dst / rel)
    assert not [p.name for p in dst.rglob('*') if p.name.startswith('_')], 'staging failed'
    return dst


def point_chromium_at_host_manifest():
    """Chromium discovers native hosts under its own registry key — add it
    alongside Google Chrome's, both pointing at the same generated manifest."""
    manifest_path = str(REPO / 'native_host' / f'{HOST_NAME}.json')
    for root in (r'Software\Google\Chrome\NativeMessagingHosts',
                 r'Software\Chromium\NativeMessagingHosts'):
        key = winreg.CreateKey(winreg.HKEY_CURRENT_USER, rf'{root}\{HOST_NAME}')
        winreg.SetValueEx(key, None, 0, winreg.REG_SZ, manifest_path)
        winreg.CloseKey(key)
    return manifest_path


def main():
    with sync_playwright() as p:
        ext_dir = stage_clean_copy()
        point_chromium_at_host_manifest()

        # Stage 1: preconditions
        check('server offline before test (else auto-start path is not exercised)',
              find_server_pid() is None, str(find_server_pid()))

        context = p.chromium.launch_persistent_context(
            user_data_dir=str(REPO / 'tmp' / 'pw-profile-chromium'),
            headless=False,
            ignore_default_args=['--disable-extensions'],
            args=[
                f'--disable-extensions-except={ext_dir}',
                f'--load-extension={ext_dir}',
                '--no-first-run',
                '--no-default-browser-check',
            ],
        )
        try:
            sws = context.service_workers
            check('extension service worker present', len(sws) == 1,
                  f'{[w.url for w in sws]}')
            if sws:
                check('service worker URL carries key-pinned extension ID',
                      sws[0].url.split('/')[2] == EXT_ID,
                      sws[0].url.split('/')[2])

            # Stage 2: panel opens -> auto-start -> online
            page0 = context.new_page()
            attach_console(page0, 'panel')
            page0.goto(f'chrome-extension://{EXT_ID}/panel.html', timeout=15000)
            panel = page0
            status = wait_for_status(panel, [ONLINE_TEXT, 'offline'], 180)
            check('panel reaches Server online via auto-start', ONLINE_TEXT in status,
                  status.strip())

            try:
                import json
                import urllib.request
                with urllib.request.urlopen(HEALTH_URL, timeout=3) as r:
                    health = json.loads(r.read())
                check('server /health after auto-start', health.get('status') == 'ok',
                      json.dumps(health))
            except Exception as e:
                check('server /health after auto-start', False, repr(e))

            # Stage 3: kill server -> 3-strike offline -> auto-revive
            import subprocess
            pid = find_server_pid()
            if pid:
                subprocess.run(['taskkill', '/F', '/PID', str(pid)], capture_output=True)
                time.sleep(2)
            check('server killed while panel online', find_server_pid() is None, f'pid {pid}')

            # 3-strike: ~3 heartbeats (45 s) + poll before the panel declares offline
            wait_for_status(panel, ['offline'], 180)
            status = wait_for_status(panel, [ONLINE_TEXT], 300)
            check('panel flips offline then auto-revives -> Server online',
                  ONLINE_TEXT in status, status.strip())

            # Stage 4: full pipeline — screenshot -> OCR -> TTS -> back to idle
            panel.bring_to_front()
            IDLE = 'Read My Screen'
            btn = panel.locator('#btn-read')
            def btn_text():
                return btn.inner_text(timeout=5000)
            panel.locator('#btn-read').click()
            t0 = time.monotonic()
            deadline = t0 + 240
            left_idle = False
            while time.monotonic() < deadline:
                txt = btn_text()
                if txt != f'📸 {IDLE}':
                    left_idle = True
                    print(f'    [{time.strftime("%H:%M:%S")}] button: {txt!r}')
                    time.sleep(5)
                elif left_idle:
                    break
                time.sleep(2)
            time.sleep(1)
            err_visible = panel.locator('#error').is_visible()
            err_text = panel.locator('#error').inner_text() if err_visible else ''
            check('Read My Screen: full pipeline runs without error',
                  left_idle and not err_visible, f'error={err_text!r}')
        finally:
            context.close()

    print()
    print(f'{sum(1 for _, ok, _ in results if ok)}/{len(results)} checks passed')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())