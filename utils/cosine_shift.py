from __future__ import division

import numpy as np

from chainer.training import extension


class CosineShift(extension.Extension):


    def __init__(self, attr, epoch=None, max_iteration=None, init=None, min_lr=0., target=None, optimizer=None):
        self._attr = attr
        self._max_iteration = max_iteration
        self._init = init
        self._target = target
        self._optimizer = optimizer
        self._t = 0
        self._last_value = None
        self._min_lr = min_lr
        self.epoch = epoch
            

    def initialize(self, trainer):
        optimizer = self._get_optimizer(trainer)
        iterator = self._get_iterator(trainer)
        if self._max_iteration is None:
            self._max_iteration = self.epoch*len(iterator.dataset)//iterator.batch_size
        # ensure that _init is set
        if self._init is None:
            self._init = getattr(optimizer, self._attr)

        if self._last_value is not None:  # resuming from a snapshot
            self._update_value(optimizer, self._last_value)
        else:
            self._update_value(optimizer, self._init)

    def __call__(self, trainer):
        self._t += 1
    
        optimizer = self._get_optimizer(trainer)
        value = self._min_lr + 0.5 * (self._init - self._min_lr) * (1 + np.cos(np.pi*self._t/self._max_iteration))
        self._update_value(optimizer, value)

    def serialize(self, serializer):
        self._t = serializer('_t', self._t)
        self._last_value = serializer('_last_value', self._last_value)
        if isinstance(self._last_value, numpy.ndarray):
            self._last_value = self._last_value.item()

    def _get_optimizer(self, trainer):
        return self._optimizer or trainer.updater.get_optimizer('main')
    
    def _get_iterator(self, trainer):
        return trainer.updater.get_iterator('main')

    def _update_value(self, optimizer, value):
        setattr(optimizer, self._attr, value)
        self._last_value = value