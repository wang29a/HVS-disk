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
SKIP_DIRECTORIES = {'__pycache__', 'variant', 'bin', 'build', 'scratch',
                    'snapshots', 'workloads', 'weight-scan', 'runtime',
                    'stage2', 'stage2-v2', 'review-20261113'}


def source_files():
    # Prune runtime trees before descent: node3 can contain full Python wheels,
    # graph checkpoints and thousands of timing points beside these sources.
    for directory, names, files in os.walk(str(ROOT), followlinks=False):
        names[:] = sorted(name for name in names if name not in SKIP_DIRECTORIES)
        for name in sorted(files):
            yield Path(directory) / name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--unit', action='store_true', help='also run existing small NumPy/protocol tests')
    args = parser.parse_args()
    manifests = ['source-manifest.json', 'research-manifest-20261114.json',
                 'research-manifest-20261116.json']
    identities = {}
    for name in manifests:
        manifest = json.loads((ROOT / name).read_text())
        for record in manifest['files']:
            relative = Path(record['file'])
            if relative.is_absolute() or '..' in relative.parts:
                raise RuntimeError('unsafe manifest path: ' + str(relative))
            data = (ROOT / relative).read_bytes()
            if record.get('append_only'):
                checked = data[:record['bytes']]
                size_ok = len(data) >= record['bytes']
            else:
                checked = data
                size_ok = len(data) == record['bytes']
            if not size_ok or hashlib.sha256(checked).hexdigest() != record['sha256']:
                raise RuntimeError('source identity changed: ' + str(relative))
        identities[name] = len(manifest['files'])
    python_count = shell_count = 0
    for path in source_files():
        if path.suffix == '.py':
            ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
            python_count += 1
        elif path.suffix == '.sh':
            subprocess.run(['bash', '-n', str(path)], check=True)
            shell_count += 1
    for directory, marker in [
        ('backlink-range-audit-20261115', 'FREEZE.json'),
        ('backlink-ab-20261115', 'FREEZE.json'),
        ('backlink-ab-20261115', 'FUNCTION-FREEZE.json'),
        ('edbt-evidence-20261115', 'FREEZE.json'),
    ]:
        frozen = json.loads((ROOT / directory / marker).read_text(encoding='utf-8'))
        data = (ROOT / directory / 'README.md').read_bytes()
        if hashlib.sha256(data[:frozen['bytes']]).hexdigest() != frozen['sha256']:
            raise RuntimeError('frozen protocol prefix changed: ' + directory + '/' + marker)
    print('PASS: {} archived, {} earlier research, {} new identities; {} Python and {} shell syntax checks; 4 frozen prefixes'.format(
        identities['source-manifest.json'], identities['research-manifest-20261114.json'],
        identities['research-manifest-20261116.json'], python_count, shell_count), flush=True)
    if args.unit:
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        for directory, names in [
            ('metric-consistent-20261111', ['test_numa_guard', 'test_queue_guard']),
            ('laion-six-cell-20261111', ['test_protocol']),
            ('u-shape-20261113', ['test_prepare_data', 'test_stage2', 'test_review']),
            ('multigraph-20261113', ['test_protocol', 'test_layers']),
            ('paper-priorities-20261114', ['test_priorities']),
            ('backlink-ab-20261115', ['test_function_tools']),
        ]:
            subprocess.run([sys.executable, '-B', '-m', 'unittest'] + names,
                           cwd=str(ROOT / directory / 'scripts'), env=env, check=True)


if __name__ == '__main__':
    main()
