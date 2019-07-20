#!/usr/bin/env python
# coding: utf-8

# # MNIST Baseline

# In[ ]:


import chainer


# ## load datasets

# In[ ]:


train, test = chainer.datasets.get_mnist(ndim=3, scale=1.0)


# ## make iterators

# In[ ]:


train_iter = chainer.iterators.SerialIterator(train, 128)
test_iter = chainer.iterators.SerialIterator(test, 128, False, False)


# ## create model

# In[ ]:


from models.lenet5 import LeNet5
from chainer import links as L

net = LeNet5()
model = L.Classifier(net)


# ## create optimizer

# In[ ]:


optimizer = chainer.optimizers.MomentumSGD(lr=0.01)
optimizer.setup(model)

optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(5e-4))


# ## create updater

# In[ ]:


updater = chainer.training.StandardUpdater(train_iter, optimizer, device=0)


# ## setting training extensions

# In[ ]:


from utils.cosine_shift import CosineShift

trainer = chainer.training.Trainer(updater, (50, 'epoch'), out='results/tmp')
trainer.extend(chainer.training.extensions.LogReport())
trainer.extend(chainer.training.extensions.Evaluator(test_iter, model, device=0), name='validation')
trainer.extend(chainer.training.extensions.observe_lr())
trainer.extend(CosineShift('lr', 50), trigger=(1, 'iteration'))


# ## training start

# In[ ]:


with chainer.using_config('autotune', True):
    trainer.run()

