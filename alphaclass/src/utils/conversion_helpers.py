import torch
import torchvision
from torch import nn
import json

class FlexibleModel:
    def __init__(self, configs, model):
        self.configs = configs
        if type(self.configs) == str:
            self.configs = self.load_config(self.configs)

        self.model = model
        if torch.cuda.is_available():
            self.model = self.model.cuda().eval().half()
        else:
            self.model =self.model.eval()

        self.sample_input = torch.randn(1, 3, self.configs['image_training_height'], self.configs['image_training_width']).cpu()
        if torch.cuda.is_available():
            self.sample_input = self.sample_input.cuda().half()


    def load_config(self, cfg):
        with open(cfg, 'r') as f:
            d = json.load(f)
        return d


    def onnx_export(self):
        self


    def to_trace(self):
        self.model = torch.jit.trace(self.model, self.sample_input)
        return self.model


    def to_script(self):
        self.model = torch.jit.script(self.model)
        return self.model
