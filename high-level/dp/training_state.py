"""Resumable shuffled batches and full stochastic training state (no frame cache)."""
import random

import numpy as np
import torch


class ResumableBatchSampler:
    def __init__(self, length, batch_size, seed=42):
        self.length = int(length)
        self.batch_size = int(batch_size)
        self.usable = self.length // self.batch_size * self.batch_size
        if not self.usable:
            raise ValueError('No complete training batch')
        self.generator = torch.Generator().manual_seed(seed)
        self.order = None
        self.cursor = 0
        self.epoch = 0

    def __len__(self):
        return self.usable // self.batch_size

    def __iter__(self):
        if self.order is None or self.cursor >= self.usable:
            self.order = torch.randperm(self.length, generator=self.generator)
            self.cursor = 0
            self.epoch += 1
        while self.cursor < self.usable:
            batch = self.order[self.cursor:self.cursor + self.batch_size].tolist()
            self.cursor += self.batch_size
            yield batch

    def state_dict(self):
        return dict(length=self.length, batch_size=self.batch_size, order=self.order,
                    cursor=self.cursor, epoch=self.epoch, generator=self.generator.get_state())

    def load_state_dict(self, value):
        if value['length'] != self.length or value['batch_size'] != self.batch_size:
            raise ValueError('Resume dataset/batch size changed')
        order = value['order']
        cursor = int(value['cursor'])
        if (order is None or order.shape != (self.length,) or
                not torch.equal(torch.sort(order).values, torch.arange(self.length)) or
                cursor < 0 or cursor > self.usable or cursor % self.batch_size):
            raise ValueError('Invalid saved sample permutation/cursor')
        self.order, self.cursor, self.epoch = order, cursor, int(value['epoch'])
        self.generator.set_state(value['generator'])


def capture_rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(value):
    random.setstate(value['python'])
    np.random.set_state(value['numpy'])
    torch.set_rng_state(value['torch'])
    if value['cuda']:
        if not torch.cuda.is_available() or len(value['cuda']) != torch.cuda.device_count():
            raise ValueError('Resume CUDA device count changed')
        torch.cuda.set_rng_state_all(value['cuda'])
