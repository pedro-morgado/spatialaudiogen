class CustomModel(object):
    def __init__(self):
        pass

    def inference_ops(self, inputs, is_training=None, reuse=False):
        raise NotImplementedError

    def loss_ops(self, logits, targets):
        raise NotImplementedError

    def evaluation_ops(self, logits, targets):
        raise NotImplementedError
