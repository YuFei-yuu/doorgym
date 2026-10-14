import unittest
from pathlib import Path
from stop_tail_diagnostics import select_windows
from stop_tail_online import smoothed,metric_rows
import tempfile
import json


class ResumeMonitoringTests(unittest.TestCase):
    def test_new_schedule_keeps_old1000_cap_unchanged(self):
        import small_forward_v3
        from stop_tail_resume5000 import train_command
        self.assertEqual(small_forward_v3.CHECKPOINTS,[250,500,1000])
        with self.assertRaises(ValueError):small_forward_v3.train_command(Path('/tmp/unused'),5000,1000)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            (root/'baseline_config.json').write_text(json.dumps(dict(root='/old/readonly/data',repo_id='local/small_forward_v3')))
            (root/'plan.json').write_text(json.dumps(dict(starting_checkpoint='/old/step_001000.pt')))
            first=train_command(root,2000,1000)
            self.assertEqual(first[first.index('--root')+1],'/old/readonly/data')
            self.assertEqual(first[first.index('--resume')+1],'/old/step_001000.pt')
            self.assertNotIn('--wandb',first)
            for target,previous in [(1000,1000),(2000,0),(4000,3000),(6000,5000)]:
                with self.assertRaises(ValueError):train_command(root,target,previous)

    def test_replan_window_does_not_use_later_stop_observation(self):
        windows=select_windows(1295,list(range(0,2500,16)),1414)
        self.assertEqual(windows[0]['center'],1280)
        self.assertEqual(windows[0]['stop_offset'],15)
        self.assertEqual(windows[0]['history_last'],1280)
        self.assertEqual(windows[-1]['center'],1408)
        self.assertTrue(all(w['history_last']==w['center'] for w in windows))

    def test_no_missing_replan_or_tail_extension(self):
        with self.assertRaises(ValueError):select_windows(3,[16,32],20)
        self.assertEqual([v['center'] for v in select_windows(18,[0,16,32,48],40)],[16,32])

    def test_smoothing_preserves_raw_values(self):
        rows=[{'train/step':10,'train/loss':1.},{'train/step':20,'train/loss':0.}]
        out=smoothed(rows)
        self.assertEqual(out[-1]['train/loss_ema'],.9)
        self.assertEqual(rows[-1]['train/loss'],0.)

    def test_partial_jsonl_is_deferred_and_conflicts_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            a=Path(d)/'a';b=Path(d)/'b'
            row={'train/step':10,'train/loss':.1,'train/lr':.0001,'train/elapsed_sec':1.}
            a.write_text(json.dumps(row)+'\n{"unfinished":')
            self.assertEqual(metric_rows([a]),[row])
            b.write_text(json.dumps(dict(row,**{'train/loss':.2}))+'\n')
            with self.assertRaises(ValueError):metric_rows([a,b])


if __name__=='__main__':unittest.main()
