import torch
from common.loss import mpjpe


def train(pipeline=None, epochs=100, dataset=None, config=None):
    train_generator = dataset.get_generator(train=True, **config.dataset_args)
    test_generator = dataset.get_generator(train=False, **config.dataset_args)

    for epoch in range(epochs):



def _train_normal(pipeline, train_gen, test_gen):
    with torch.set_grad_enabled(True):
        for x_batch, y_batch in train_gen():
            y_pred = pipeline(x_batch)
            loss = mpjpe(y_pred, y_batch)

            loss.backward()
