"""Short checks for matching, shard coverage, and the exact scanner."""

import itertools
import os
from pathlib import Path
import struct
import subprocess
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from clustering import balanced_assign, bit, coverage, expand, greedy, member_summary, optimal, select
from run_phase0 import WEIGHTS


class PairedPartitionTests(unittest.TestCase):
    def test_balanced_training_respects_quotas(self):
        costs = np.arange(51).reshape(17, 3)
        self.assertEqual(np.bincount(balanced_assign(costs)).tolist(), [6, 6, 5])

    def test_max_overlap_can_improve_on_greedy(self):
        overlap = np.array([[9, 8, 0], [8, 0, 0], [0, 0, 7]])
        columns = np.arange(3)
        score = lambda p: overlap[list(p), columns].sum()
        self.assertEqual(score(greedy(overlap)), 16)
        result = optimal(overlap)
        self.assertEqual(score(result), max(map(score, itertools.permutations(columns))))
        self.assertEqual(score(result), 23)

    def test_effective_boundary_memberships_and_coverage(self):
        x = np.array([0, 0, 1, 1], dtype="<u4")
        y = np.array([0, 1, 1, 2], dtype="<u4")
        mask = bit(x) | bit(y)
        self.assertEqual(member_summary(mask, 3)["membership_histogram"], [2, 2, 0, 0])
        new, eps, changed = expand(mask, (bit(np.array([1, 0, 2, 0], "<u4")),
                                         bit(np.array([2, 2, 0, 0], "<u4"))),
                                   (np.array([.1,.02,.3,.4]), np.array([.5,.2,.4,.2])), .25)
        self.assertEqual(changed, 1)
        self.assertEqual(eps, .1)
        self.assertEqual(member_summary(new, 3)["point_copies"], 7)
        query = np.array([0])
        nbr = np.tile(np.array([[[1, 2, 3]]]), (1, len(WEIGHTS), 1))
        self.assertAlmostEqual(coverage(mask,query,nbr)[0]["mean"],1/3)
        self.assertEqual(coverage(new,query,nbr)[0]["mean"],1)

    def test_same_case_must_clear_work_and_every_weight(self):
        gates = dict(pair_rate=.5, max_work_ratio=1.6, coverage=.95)
        run = dict(k=8, methods=dict(C3=dict(pair_rate=.61,cases={
            "0.0":dict(work_ratio=1.39, expansion_target=0.,coverage=[{"mean":a} for a in [.95,.95,.94,.96,.96]]),
            "0.1":dict(work_ratio=1.61, expansion_target=.1,coverage=[{"mean":.96}]*5)})))
        self.assertIsNone(select([run],gates))
        run["methods"]["C3"]["cases"]["0.1"]["work_ratio"] = 1.6
        self.assertEqual(select([run],gates)["rate"],.1)

    @unittest.skipUnless(os.environ.get("PAIRED_EXACT_BINARY"), "set the compiled exact scanner")
    def test_exact_fusion_stable_ties_and_checkpoint_resume(self):
        rng = np.random.RandomState(80)
        x,y = [rng.uniform(-2,2,(173,5)).astype("<f4") for _ in range(2)]
        x[0],y[0] = x[1],y[1]
        ids = rng.permutation(len(x))[:140].astype("<u4")
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name,values in zip(("x","y"),(x,y)):
                with (root/name).open("wb") as f:
                    f.write(struct.pack("<II",*values.shape))
                    values.tofile(f)
            ids.tofile(root/"ids")
            out = root/"out"
            command = [os.environ["PAIRED_EXACT_BINARY"]] + [str(root/k) for k in ("x","y","ids","out")]
            subprocess.run(command,check=True)
            raw = out.read_bytes()
            rows = np.frombuffer(raw,"<u4",offset=12).reshape(len(ids),len(WEIGHTS),40)
            for i,id0 in enumerate(ids):
                a = np.linalg.norm(x.astype("f8") - x[id0],axis=1)
                b = np.linalg.norm(y.astype("f8") - y[id0],axis=1)
                for j,w in enumerate(WEIGHTS):
                    mix = w*a + (1-w)*b
                    mix[id0] = np.inf
                    self.assertEqual(rows[i,j].tolist(),np.argsort(mix,kind="stable")[:40].tolist())
            out.write_bytes(raw[:12+74*5*40*4+3])
            subprocess.run(command,check=True)
            self.assertEqual(out.read_bytes(),raw)
