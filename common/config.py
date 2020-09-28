class Config():
    # Hyper params
    # Generator paramas
    # Model structure params
    # names
    def __init__(self):
        self.epoch = 100
        self.lr = 1e-3
        self.lr_decay = [0.95]
        self.lr_decay_step = [1]
        self.use_amp = True

        self.batch_size = 32
        self.length = 1
        self.chunked = True
        self.keypoint = 'cpn'
        self.normalized = True
        self.padding = True
        self.receptive_field = 243
        self.cuda = True

        self.pre_refienment_args = None
        self.pos_estimator_args = None
        self.post_refienment_args = None
