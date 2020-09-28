from common.dataset import H36MDataset
from common.signal_processing import ExponentialMovingAverage

dataset = H36MDataset()
dataset.get_generator('cpn', padding=True, length=1, chunked=True, receptive_field=243, preprocessor=ExponentialMovingAverage(0.9))

# lr reset by epoch w/ read() func