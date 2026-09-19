#!/usr/bin/env python3
"""Pixion patch: redact SGLang serving credentials from diagnostics.

SGLang v0.5.20 logs ``ServerArgs.resolved_dict()`` at startup and returns the
same projection from ``/server_info``.  It also returns ``launch_command``,
which contains the literal CLI values.  Redact the three credential-bearing
fields without changing the live ``ServerArgs`` values used by authentication.

This is a narrow backport of the security boundary proposed upstream in
sgl-project/sglang#39449.  It is idempotent and fails closed if the pinned
source anchors change.
"""

import glob
import importlib.util
import pathlib
import sys
import sysconfig


MARKER = "pixion-secret-redaction-v1"


def find_file(rel: str) -> pathlib.Path:
    candidates: list[pathlib.Path] = []
    try:
        spec = importlib.util.find_spec("sglang")
        if spec is not None and spec.submodule_search_locations:
            candidates.extend(pathlib.Path(p) for p in spec.submodule_search_locations)
    except Exception:
        pass
    for key in ("purelib", "platlib"):
        try:
            candidates.append(pathlib.Path(sysconfig.get_paths()[key]) / "sglang")
        except Exception:
            pass
    for pattern in (
        "/opt/*/lib/python3*/site-packages/sglang",
        "/usr/local/lib/python3*/site-packages/sglang",
        "/usr/lib/python3*/site-packages/sglang",
        "/sgl-workspace/*/python/sglang",
    ):
        candidates.extend(pathlib.Path(p) for p in glob.glob(pattern))
    seen: set[pathlib.Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        path = candidate / rel
        if path.is_file():
            return path
    raise FileNotFoundError(f"cannot locate sglang/{rel} in {sorted(map(str, seen))}")


def main() -> int:
    path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else find_file("srt/server_args.py")
    source = path.read_text()
    if MARKER in source:
        print(f"[pixion] already patched: {path}")
        return 0

    launch_old = '        return getattr(self, "_launch_command", None)\n'
    launch_new = '''        # pixion-secret-redaction-v1: never publish credentials embedded in argv.
        command = getattr(self, "_launch_command", None)
        if command is None:
            return None
        for field_name in ("api_key", "admin_api_key", "ssl_keyfile_password"):
            value = resolution_result(self, field_name)
            values = value if isinstance(value, (list, tuple, set)) else (value,)
            for secret in values:
                if secret:
                    command = command.replace(str(secret), "<redacted>")
        return command
'''
    if source.count(launch_old) != 1:
        raise SystemExit("[pixion] launch_command anchor count != 1; aborting")
    source = source.replace(launch_old, launch_new)

    resolved_old = '''        return {
            field.name: _plain(resolution_result(self, field.name))
            for field in record_fields(type(self))
        }
'''
    resolved_new = '''        result = {
            field.name: _plain(resolution_result(self, field.name))
            for field in record_fields(type(self))
        }
        # Keep the operational record untouched; redact only diagnostic output.
        for field_name in ("api_key", "admin_api_key", "ssl_keyfile_password"):
            value = result.get(field_name)
            if value is not None:
                count = len(value) if isinstance(value, (list, tuple, set)) else 1
                result[field_name] = f"<redacted:{count} value{'s' if count != 1 else ''}>"
        return result
'''
    if source.count(resolved_old) != 1:
        raise SystemExit("[pixion] resolved_dict anchor count != 1; aborting")
    source = source.replace(resolved_old, resolved_new)

    compile(source, str(path), "exec")
    path.write_text(source)
    print(f"[pixion] redacted SGLang credentials in logs and server-info: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
