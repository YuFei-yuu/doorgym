"""Expert action replay variants. No learned policy or training is launched."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

# wholebody_float imports Isaac Gym before Torch, as required by the Gym runtime.
import wholebody_float as backend
import numpy as np
import torch
from diagnostic_contract import clipped_legacy_action, legacy_contract, validate_checkpoint_contract


def replay_configuration(mode, quaternion_mode='source'):
    return dict(restore_freeze=mode == 'recorded',
                restore_quaternion=mode == 'recorded' and quaternion_mode == 'source')


class DiagnosticReplay(backend.WholebodyFloat):
    diagnostic_options = None

    @classmethod
    def configure(cls, args):
        super().configure(args)
        if not args.float_replay or args.dp_policy_checkpoint:
            raise ValueError('Diagnostics require expert replay and prohibit a policy checkpoint')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        options = self.diagnostic_options
        self.original_actions = self.replay.copy()
        self.original_flags = [(r['freeze_arm_default'], r['freeze_arm_zero']) for r in self.replay_rows]
        if len(self.replay) != len(self.replay_rows) or self.args.steps != len(self.replay):
            raise ValueError('Replay must run exactly the recorded expert horizon')
        contract = None
        variant = replay_configuration(options.mode, getattr(options, 'quaternion_mode', 'source'))
        if not variant['restore_quaternion']:
            self.replay_quaternions = None
        if not variant['restore_freeze']:
            # The policy applicator clears these flags and normalizes the raw
            # target quaternion. Do not reinstate either expert side channel.
            for row in self.replay_rows:
                row['freeze_arm_default'] = False
                row['freeze_arm_zero'] = False
        if options.mode == 'clipped':
            ckpt = torch.load(options.stats_checkpoint, map_location='cpu')
            validate_checkpoint_contract(ckpt['config'])
            contract = legacy_contract(ckpt['stats'])
            self.replay = clipped_legacy_action(self.replay, contract)
        source = Path(__file__)
        info = dict(mode=options.mode, original_horizon=len(self.replay),
                    stats_checkpoint=options.stats_checkpoint, action_contract=contract,
                    autonomous_policy=False, restores_freeze_flags=variant['restore_freeze'],
                    restores_executed_quaternion=variant['restore_quaternion'],
                    quaternion_mode=getattr(options, 'quaternion_mode', 'source'),
                    preserves_natural_snap=True, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
        (self.output/'diagnostic.json').write_text(json.dumps(info, indent=2))

    def before_step(self, *args, **kwargs):
        super().before_step(*args, **kwargs)
        i = self.record['step']
        self.record['diagnostic'] = dict(mode=self.diagnostic_options.mode,
            expert_action=self.original_actions[i].tolist(), requested_action=self.replay[i].tolist(),
            expert_freeze_arm_default=self.original_flags[i][0], expert_freeze_arm_zero=self.original_flags[i][1],
            autonomous_policy=False)
        # Keep recording the full expert horizon even if pass_done is latched.
        self.task_done = False

    def close(self, error=None):
        if self.closed:
            return
        super().close(error)
        path = self.output/'summary.json'
        if path.exists():
            value = json.loads(path.read_text())
            value.update(diagnostic_replay_mode=self.diagnostic_options.mode, autonomous_policy=False,
                         recorded_horizon=len(self.replay),
                         replay_exhausted_normally=error is None and value['frames'] == len(self.replay))
            if error is None and not value['pass_done'] and value['frames'] == len(self.replay):
                value['end_reason'] = 'replay_exhausted'
            path.write_text(json.dumps(value, indent=2))


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--diagnostic-mode', dest='mode', choices=['recorded', 'actions', 'clipped'], required=True)
    parser.add_argument('--stats-checkpoint')
    parser.add_argument('--diagnostic-quaternion-mode', dest='quaternion_mode', choices=['source', 'action'], default='source',
                        help='action applies identical original 10D quaternion normalization to both freeze variants')
    options, remaining = parser.parse_known_args()
    if options.mode == 'clipped' and not options.stats_checkpoint:
        parser.error('Clipped replay requires --stats-checkpoint')
    DiagnosticReplay.diagnostic_options = options
    sys.argv = [sys.argv[0]]+remaining
    try:
        backend.upstream.main(backend=DiagnosticReplay)
    except BaseException as exc:
        if DiagnosticReplay.instance is not None:
            DiagnosticReplay.instance.close(error=repr(exc))
        raise


if __name__ == '__main__':
    main()
