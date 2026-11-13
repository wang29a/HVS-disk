"""Freeze the tuning/report split using existing OpenImages queries and GT."""
import hashlib
import json
import os
from pathlib import Path
import struct

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT/'tuning-v2'
SEED = 20261111


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    protocol = json.loads((ROOT/'analysis/protocol.json').read_text())
    source = protocol['source']
    keys = ('query1','query2','alpha','gt')
    matrices = {}
    hashes = {}
    for key in keys:
        path = Path(source[key])
        hashes[key] = digest(path)
        if hashes[key] != source['input_sha256'][key]:
            raise RuntimeError('original query or GT changed: '+key)
        n, d = struct.unpack('<II',path.read_bytes()[:8])
        if n!=1000 or path.stat().st_size!=8+n*d*4:
            raise RuntimeError('unexpected matrix shape: '+key)
        matrices[key] = np.fromfile(str(path),dtype='<u4',offset=8).reshape(n,d)
    ids = np.random.RandomState(SEED).permutation(1000)
    manifest = dict(seed=SEED,source_hashes=hashes,cohorts={},
                    purpose='tuning decisions use tune only; report IDs stay held out for configuration selection')
    outputs = {}
    for name, selected in (('tune',ids[:500]),('report',ids[500:])):
        folder = OUT/name
        cohort = dict(query_ids=[int(i) for i in selected],files={})
        for key, matrix in matrices.items():
            path = folder/(key+'.bin')
            payload = struct.pack('<II',500,matrix.shape[1])+matrix[selected].tobytes()
            outputs[path] = payload
            cohort['files'][key] = dict(path=str(path),sha256=hashlib.sha256(payload).hexdigest())
        weights = matrices['alpha'][selected].view('<f4').reshape(-1)
        cohort['alpha_bin_queries'] = np.bincount(np.minimum(9,(weights*10).astype(int)),
                                                 minlength=10).tolist()
        manifest['cohorts'][name] = cohort
    a,b = (set(c['query_ids']) for c in manifest['cohorts'].values())
    if a&b or a|b!=set(range(1000)):
        raise RuntimeError('query split is not disjoint and complete')
    path = OUT/'split.json'
    if path.exists() and json.loads(path.read_text())!=manifest:
        raise RuntimeError('refusing to alter a frozen query split')
    for output,payload in outputs.items():
        if output.exists() and output.read_bytes()!=payload:
            raise RuntimeError('existing split bytes changed: '+str(output))
    for output,payload in outputs.items():
        output.parent.mkdir(parents=True,exist_ok=True)
        temp = output.with_suffix('.partial')
        temp.write_bytes(payload)
        os.replace(str(temp),str(output))
    temp = path.with_suffix('.partial')
    temp.write_text(json.dumps(manifest,indent=2)+'\n')
    os.replace(str(temp),str(path))
    print('Frozen 500 tuning / 500 reporting queries; existing GT IDs preserved')


if __name__=='__main__':
    main()
