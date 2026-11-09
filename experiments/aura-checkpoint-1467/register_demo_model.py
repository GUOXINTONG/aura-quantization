#!/usr/bin/env python3
"""Add one main model to aura_model_registry.json (idempotent, atomic, with a backup)."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("registry")
    parser.add_argument("--id", required=True)
    parser.add_argument("--name")
    parser.add_argument("--path", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    path = Path(args.registry)
    registry = json.loads(path.read_text(encoding="utf-8"))
    entry = {"id": args.id, "name": args.name or args.id, "path": args.path,
             "gpu": str(args.gpu), "port": args.port}
    existing = next((m for m in registry["main_models"] if m["id"] == args.id), None)
    if existing == entry:
        print("already registered:", entry)
        return 0
    if existing is not None:
        raise SystemExit(f"id {args.id} is registered with different settings: {existing}")
    clash = [m["id"] for m in registry["main_models"] if int(m["port"]) == args.port]
    if clash:
        raise SystemExit(f"port {args.port} already used by {clash}")
    backup_dir = path.parent / "backup"
    backup_dir.mkdir(exist_ok=True)
    shutil.copy2(path, backup_dir / f"{path.name}.bak-{datetime.now():%Y%m%d-%H%M%S}")
    registry["main_models"].append(entry)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".registry.")
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(registry, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.chmod(tmp, path.stat().st_mode & 0o777)
    os.replace(tmp, path)
    print("registered:", entry)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
