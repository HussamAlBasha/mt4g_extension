"""Small mathematical checks of the RQ3 reducer (standard-library unittest)."""
import itertools
import math
import random
import statistics
import unittest
import compare_results as c
from generate_orders import generate
from collections import Counter

class StatisticalChecks(unittest.TestCase):
    def test_tied_ranks_and_constant_vectors(self):
        self.assertEqual(c.ranks([4,1,1,9]),[3,1.5,1.5,4])
        self.assertTrue(math.isnan(c.rank_stability([{0:1,1:1}]*12,[0,1])))
        self.assertEqual(c.rank_stability([{0:1,1:2}]*12,[0,1]),1)
        self.assertEqual(c.rank_stability([{0:1,1:2},{0:2,1:1}],[0,1]),-1)

    def test_holm_known_family(self):
        actual=c.holm_adjust([.04,.001,.03,.2])
        for a,b in zip(actual,[.09,.004,.09,.2]):
            self.assertAlmostEqual(a,b)

    def test_statistic_matches_centered_between_device_ss(self):
        sweeps=[{0:1,1:2,2:6},{0:2,1:3,2:4}]
        for permutation in itertools.permutations(sweeps[1].values()):
            rows=[sweeps[0],dict(enumerate(permutation))]
            means=[statistics.mean(row[d] for row in rows) for d in range(3)]
            grand=statistics.mean(means)
            self.assertAlmostEqual(c.effect_statistic(rows,[0,1,2])-3*grand**2,sum((m-grand)**2 for m in means))

    def test_permutation_against_exact_small_null(self):
        sweeps=[dict(enumerate([-1,0,1]))]*2
        observed=c.effect_statistic(sweeps,[0,1,2])
        all_rows=list(itertools.permutations([-1,0,1]))
        exact=sum(c.effect_statistic([dict(enumerate(a)),dict(enumerate(b))],[0,1,2])>=observed-1e-15 for a in all_rows for b in all_rows)/36
        self.assertAlmostEqual(exact,1/6)
        self.assertAlmostEqual(c.permutation_p_value(sweeps,[0,1,2],9999,'exact-test'),exact,delta=.015)
        self.assertEqual(c.permutation_p_value([{0:0,1:0}]*12,[0,1],99,'constant'),1)

    def test_balance(self):
        records=generate(12,12,'11360825','vipa1008')
        self.assertEqual(Counter((r['device'],r['position']) for r in records),Counter({(d,p):1 for d in range(12) for p in range(12)}))
        orders=[[r['device'] for r in records if r['repeat']==s] for s in range(1,13)]
        self.assertEqual(Counter(p for row in orders for p in zip(row,row[1:])),Counter({(a,b):1 for a in range(12) for b in range(12) if a!=b}))

if __name__=='__main__':
    unittest.main()
