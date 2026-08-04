class SGDContinual:

    def before_task(self, model):
        pass

    def loss(self, model, ce_loss):
        return ce_loss

    def after_task(self, model):
        pass