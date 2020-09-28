import yaml
from common.config import Config


def get_configs():
    f = open('config.yaml')
    raw_cfg = yaml.load(f, Loader=yaml.FullLoader)
    f.close()
    print(raw_cfg)



if __name__ == '__main__':
    get_configs()