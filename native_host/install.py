#!/usr/bin/env python3
"""One-time installer for the TTS native messaging host.

Chrome finds a native host only through a host manifest JSON with absolute
paths plus an HKCU registry value pointing at it (mandatory on Windows), and
the .bat wrapper binds a concrete Python interpreter — all machine-specific,
so none can be committed. This script generates the two files and registers
the registry value; run again with --uninstall to remove all three.

Usage:
    python native_host/install.py [--extension-id ID] [--python PATH]
    python native_host/install.py --uninstall

No admin rights: everything lives under HKEY_CURRENT_USER.
"""

import argparse
import json
import sys
import winreg
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tts_native_host import HOST_NAME

# Pinned unpacked-extension ID (derived from the "key" field in manifest.json).
# Overridable with --extension-id; required if left empty.
DEFAULT_EXTENSION_ID = 'isdactysjoifvgcc'

# kokoro requires Python >=3.10,<3.13 (its numpy==1.26.4 pin has no cp313 wheels),
# so the server/host run under the project's Python 3.12 venv by default.
DEFAULT_PYTHON = str((Path(__file__).resolve().parents[1] / '.venv' / 'Scripts' / 'python.exe'))
REGISTRY_PATH = rf'Software\Google\Chrome\NativeMessagingHosts\{HOST_NAME}'
HERE = Path(__file__).resolve().parent


def write_manifest(extension_id: str) -> Path:
    manifest_path = HERE / f'{HOST_NAME}.json'
    manifest = {
        'name': HOST_NAME,
        'description': 'Spawns the kokoro_server.py backend for the TTS screen reader extension',
        'path': str(HERE / 'tts_native_host.bat'),
        'type': 'stdio',
        'allowed_origins': [f'chrome-extension://{extension_id}/'],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest_path


def write_bat(python_path: str) -> Path:
    bat_path = HERE / 'tts_native_host.bat'
    # -u keeps the binary length-prefixed stdio unbuffered (Chrome protocol).
    bat_path.write_text(f'@"{python_path}" -u "%~dp0tts_native_host.py"\r\n', encoding='ascii')
    return bat_path


def register(manifest_path: Path) -> None:
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, REGISTRY_PATH) as key:
        winreg.SetValueEx(key, None, 0, winreg.REG_SZ, str(manifest_path))


def is_registered() -> bool:
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, REGISTRY_PATH):
            return True
    except FileNotFoundError:
        return False


def uninstall() -> None:
    try:
        winreg.DeleteKey(winreg.HKEY_CURRENT_USER, REGISTRY_PATH)
    except FileNotFoundError:
        pass
    (HERE / f'{HOST_NAME}.json').unlink(missing_ok=True)
    (HERE / 'tts_native_host.bat').unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description='Install/uninstall the TTS native messaging host')
    parser.add_argument('--extension-id', default=DEFAULT_EXTENSION_ID,
                        help='unpacked extension ID to allow (default: pinned ID from the plan)')
    parser.add_argument('--python', default=DEFAULT_PYTHON, help='Python interpreter for the host .bat')
    parser.add_argument('--uninstall', action='store_true', help='remove registry key and generated files')
    args = parser.parse_args()

    if args.uninstall:
        uninstall()
        print(f'[install] Removed registry key HKCU\\{REGISTRY_PATH} and generated files.')
        return

    if not args.extension_id:
        parser.error('--extension-id is required (DEFAULT_EXTENSION_ID is not set)')

    manifest_path = write_manifest(args.extension_id)
    bat_path = write_bat(args.python)
    register(manifest_path)
    print(f'[install] Host manifest: {manifest_path}')
    print(f'[install] Host wrapper:  {bat_path}')
    print(f'[install] Registry:      HKCU\\{REGISTRY_PATH} -> {manifest_path}')
    print(f'[install] Allowed origin: chrome-extension://{args.extension_id}/')


if __name__ == '__main__':
    main()