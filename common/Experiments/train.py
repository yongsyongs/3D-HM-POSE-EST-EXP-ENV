import torch
import time
from torchsummary import summary
from common.loss import mpjpe

def train(cfg):
    print('Load Dataset...', end='')
    dataset = cfg.dataset(cfg)
    train_generator = dataset.get_chunked_generator(train=True) if cfg.chunked else dataset.get_unchunked_generator(train=True, shuffle=True)
    test_generator = dataset.get_unchunked_generator(train=False)
    print('done')

    pipeline = __import__('common.pipelines.' + cfg.pipeline, fromlist=[cfg.pipeline]).pipeline(dataset.num_joints, cfg)
    optimizer = cfg.optimizer(pipeline.parameters(), lr=cfg.lr, amsgrad=cfg.amsgrad)

    if cfg.amp:
        amp_scaler = torch.cuda.amp.GradScaler()

    lr = cfg.lr
    for epoch in range(cfg.epochs):
        times = []
        start_time = time.time()
        time_per_iter = 0

        pipeline.train()
        for i, (x_batch, y_batch) in enumerate(train_generator()):

            optimizer.zero_grad()
            if cfg.amp:
                with torch.cuda.amp.autocast():
                    y_pred = pipeline(x_batch, y_batch)
            else:
                with torch.set_grad_enabled(True):
                    y_pred = pipeline(x_batch, y_batch)
            loss = pipeline.get_loss()

            print('epoch %d    iter %d    loss: %.2f    %.2fsec/iter    lr:%f\t\t\t'
                  % ((epoch + 1), i, loss.data, time_per_iter, lr), end='\r')

            if cfg.amp:
                amp_scaler.scale(loss).backward()
                amp_scaler.step(optimizer)
                amp_scaler.update()
            else:
                loss.backward()
                optimizer.step()

            time_per_iter = time.time() - start_time
            times.append(time_per_iter)
            start_time = time.time()

        pipeline.eval()
        with torch.set_grad_enabled(False):
            losses = []
            for x_batch, y_batch in train_generator():
                y_pred = pipeline(x_batch, y_batch)
                losses.append(pipeline.get_loss().data)
            print('epoch %d    iter %d    loss: %.2f    %.2fsec/iter    lr:%f\t\t\t'
                  % ((epoch + 1), i + 1, torch.mean(torch.stack(losses)).data, sum(times) / len(times), lr))

            if (epoch + 1) % 5 == 0:
                losses = []
                for x_batch, y_batch in test_generator():
                    y_pred = pipeline(x_batch, y_batch)
                    losses.append(pipeline.get_loss().data)
                print('test result - loss: %.2f' % torch.mean(torch.stack(losses)).data)

        for _lr_decay, _lr_decay_step in zip(cfg.lr_decay, cfg.lr_decay_step):
            if (epoch + 1) % _lr_decay_step == 0:
                lr *= _lr_decay
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= _lr_decay