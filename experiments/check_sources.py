#!/usr/bin/env python3
"""Verify archived sources without launching experiment queues or SSH tasks."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import os

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--unit', action='store_true', help='also run existing small NumPy/protocol tests')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'source-manifest.json').read_text())
    for record in manifest['files']:
        relative = Path(record['file'])
        if relative.is_absolute() or '..' in relative.parts:
            raise RuntimeError('unsafe manifest path: ' + str(relative))
        data = (ROOT / relative).read_bytes()
        if len(data) != record['bytes'] or hashlib.sha256(data).hexdigest() != record['sha256']:
            raise RuntimeError('archived source changed: ' + str(relative))
    python_count = shell_count = 0
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file() or any(p in path.parts for p in ('__pycache__', 'variant', 'bin', 'scratch', 'snapshots', 'workloads', 'weight-scan')):
            continue
        if path.suffix == '.py':
            ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
            python_count += 1
        elif path.suffix == '.sh':
            subprocess.run(['bash', '-n', str(path)], check=True)
            shell_count += 1
    print('PASS: {} archived identities; {} Python and {} shell syntax checks'.format(
        len(manifest['files']), python_count, shell_count), flush=True)
    if args.unit:
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        for directory, names in [
            ('metric-consistent-20261111', ['test_numa_guard', 'test_queue_guard']),
            ('laion-six-cell-20261111', ['test_protocol']),
        ]:
            subprocess.run([sys.executable, '-B', '-m', 'unittest'] + names,
                           cwd=str(ROOT / directory / 'scripts'), env=env, check=True)


if __name__ == '__main__':
    main()
