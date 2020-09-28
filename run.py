from common.dataset import H36MDataset

dataset = H36MDataset()
dataset.get_generator('cpn', padding=True, length=1, chunked=True, receptive_field=27)