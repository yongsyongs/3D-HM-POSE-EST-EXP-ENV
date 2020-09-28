class Pipeline():
    def __init__(self, modules=[], pred_func=None):
        self.modules = modules

        def pred(self, x):
            for module in self.modules:
                x = module(x)
            return x

        self.pred = pred if pred_func is None else pred_func

    def set_pred(self, func):
        self.pred = func


    def __call__(self, x):
        return self.pred(self, x)