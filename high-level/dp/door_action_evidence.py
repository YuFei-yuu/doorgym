"""Observe the real single-env controller queue. Never changes a policy action.

The worker owns these counters; the parent receives a receipt with each act.
No phase, planning signal, model input or normalization is introduced.
"""
import hashlib
import os
import numpy as np


class WorkerActionEvidence:
    def __init__(self, controller):
        self.controller = controller
        self.epoch = 0
        self.chunk_id = 0
        self.emission = 0
        self.observation = 0
        self.chunk_rows = None
        self.chunk_index = 0
        self.last = None
        original = controller.sample_action_chunk

        def sampled(*args, **kwargs):
            result = original(*args, **kwargs)
            rows = np.asarray(list(controller.action_queue), dtype=np.float32)
            if rows.shape != (controller.action_horizon, controller.action_dim):
                raise RuntimeError('Incomplete real action queue after sample_action_chunk')
            self.chunk_rows = rows.copy()
            self.chunk_id += 1
            self.chunk_index = 0
            self.chunk_hash = hashlib.sha256(rows.tobytes()).hexdigest()
            self.chunk_observation = self.observation
            return result
        controller.sample_action_chunk = sampled

    def reset(self):
        self.epoch += 1
        self.chunk_id = self.emission = self.observation = self.chunk_index = 0
        self.chunk_rows = self.last = None

    def queue_state(self):
        return dict(epoch=self.epoch, action_queue_depth=len(self.controller.action_queue),
                    observation_queue_depth=len(self.controller.obs_buffer),
                    chunk_id=self.chunk_id, emitted_actions=self.emission, worker_pid=os.getpid())

    def act(self, *args, **kwargs):
        before = len(self.controller.action_queue)
        chunk_before = self.chunk_id
        self.observation += 1
        action = self.controller.act(*args, **kwargs)
        after = len(self.controller.action_queue)
        new_chunk = self.chunk_id != chunk_before
        if self.chunk_rows is None or self.chunk_index >= len(self.chunk_rows):
            raise RuntimeError('Missing worker chunk provenance for emitted action')
        if not np.array_equal(np.asarray(action, dtype=np.float32), self.chunk_rows[self.chunk_index]):
            raise RuntimeError('Emitted action differs from the actual sampled queue row')
        if after != (self.controller.action_horizon - 1 if new_chunk else before - 1):
            raise RuntimeError('Unexpected real queue transition')
        self.emission += 1
        self.last = dict(schema='door_worker_action_v1', worker_pid=os.getpid(), epoch=self.epoch,
            chunk_id=self.chunk_id, emission_seq=self.emission, observation_seq=self.observation,
            sampled_observation_seq=self.chunk_observation, action_index=self.chunk_index,
            queue_before=before, queue_after=after, new_chunk=new_chunk,
            executed_horizon=self.controller.action_horizon, chunk_sha256=self.chunk_hash,
            action=np.asarray(action, dtype=np.float32).tolist())
        # First receipt of a chunk carries the actual selected rows for independent
        # action-by-action comparison; these are observed, not reconstructed.
        if self.chunk_index == 0:
            self.last['selected_chunk_actions'] = self.chunk_rows.tolist()
        self.chunk_index += 1
        return action
