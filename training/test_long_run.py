"""Small CPU checks for expensive unattended-run safeguards, no held-out data."""
import copy
import unittest
from pathlib import Path
import json
import hashlib

from long_run import guards, RUN
from long_run_data import rows
from generated_splits import assign, group_key


class SafetyChecks(unittest.TestCase):
    def fixture(self):
        group={'drawings':1000,'top1':.8,'top3':.9}
        custom={'drawings':1000,'top1':.95,'top3':.99}
        row={'real':group,'synthetic':custom,'all':group,
             'false_alarms':{'rate':.01},'black_hole_rate':.001,
             'custom_classes':{'heart':custom}}
        baseline={m:copy.deepcopy(row) for m in ['black','filled','painted']}
        candidate=copy.deepcopy(baseline)
        for mode in ['filled','painted']:
            candidate[mode]['real']['top1']+=.02
            candidate[mode]['real']['top3']+=.02
        return candidate,{'installed':baseline,'previous_real_style':copy.deepcopy(baseline)}

    def test_heart_cannot_hide_in_aggregate(self):
        candidate,reference=self.fixture()
        self.assertTrue(guards(candidate,reference)['eligible'])
        candidate['filled']['custom_classes']['heart']['top1']=.5
        g=guards(candidate,reference)
        self.assertFalse(g['eligible'])
        self.assertIn('filled/heart/top1',g['failures'])

    def test_false_alarm_one_extra_rejected(self):
        candidate,reference=self.fixture()
        candidate['painted']['false_alarms']['rate']=.011
        self.assertIn('painted/custom-false-alarm',guards(candidate,reference)['failures'])

    def test_custom_filled_top3_is_guarded(self):
        candidate,reference=self.fixture()
        candidate['filled']['synthetic']['top3']=.90
        self.assertIn('filled/synthetic/top3',guards(candidate,reference)['failures'])

    def test_prompt_groups_and_pinned_validation(self):
        meta=json.loads((RUN/'generated-validation.json').read_text())
        val=set(meta['groups'])
        n=0
        for row in rows():
            self.assertEqual(row['split'],assign(row))
            if row['split']=='train':
                self.assertNotIn(group_key(row),val);n+=1
        self.assertGreater(n,0)
        self.assertEqual(hashlib.sha256((RUN/'generated-validation.npz').read_bytes()).hexdigest(),meta['sha256'])


if __name__=='__main__':unittest.main()
