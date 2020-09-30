import datetime
import os

from common.experiments.config import get_configs
from common.experiments.train import train

now = str(datetime.datetime.now()).split(' ')
date = now[0].replace('-', '')
time = now[1].split('.')[0].replace(':', '')[:4]
exp_path = os.path.join('results', date + time)
os.makedirs(exp_path, exist_ok=True)

print('Load config...', end='')
cfgs = get_configs()
print('done!\n')
print('Got %d configs' % len(cfgs))
if len(cfgs) > 1:
    print('Variable keys:', cfgs[0].yaml['Variable'])
print()
print('Start experiments\n')

for i, cfg in enumerate(cfgs):
    print('=' * 50, f'train #{i}', '=' * 50)
    rst_path = os.path.join(exp_path, f'opt_{i}')
    print('result will be saved in', rst_path)
    os.makedirs(rst_path, exist_ok=False)
    cfg.result_path = rst_path
    train(cfg)

print('=' * 100)
print('Done all experiment.')