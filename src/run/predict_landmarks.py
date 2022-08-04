import numpy as np
import os
import argparse
import torch
import random
import config
import json
from src.model.loss import *
from PIL import Image
import matplotlib.pyplot as plt

from src.dataset.dataloader import img_to_tensor, uv_map_to_tensor


class BasePredictor:
    def __init__(self, weight_path):
        self.model = self.get_model(weight_path)

    def get_model(self, weight_path):
        raise NotImplementedError

    def predict(self, img):
        raise NotImplementedError


class SADRNPredictor(BasePredictor):
    def __init__(self, weight_path):
        super(SADRNPredictor, self).__init__(weight_path)

    def get_model(self, weight_path):
        from src.model.SADRN import get_model
        model = get_model()
        pretrained = torch.load(weight_path, map_location=config.DEVICE)
        model_dict = model.state_dict()
        match_dict = {k: v for k, v in pretrained.items() if (k in model_dict and v.shape == model_dict[k].shape)}
        model_dict.update(match_dict)
        model.load_state_dict(model_dict)
        model = model.to(config.DEVICE)
        model.eval()
        return model

    def predict(self, img):
        # TODO normalize
        with torch.no_grad():
            out = self.model({'img': image}, {}, 'predict')
        #out['face_uvm'] *= config.POSMAP_FIX_RATE
        out['kpt_uvm'] *= config.POSMAP_FIX_RATE

        #out['face_uvm'] = out['face_uvm'].cpu().permute(0, 2, 3, 1).numpy()[0]
        out['kpt_uvm'] = out['kpt_uvm'].cpu().permute(0, 2, 3, 1).numpy()[0]
        out['offset_uvm'] = out['offset_uvm'].cpu().permute(0, 2, 3, 1).numpy()[0]
        out['attention_mask'] = out['attention_mask'].cpu().permute(0, 2, 3, 1).numpy()[0]
        return out


class SADRNv2Predictor(SADRNPredictor):
    def __init__(self, weight_path):
        super(SADRNv2Predictor, self).__init__(weight_path)

    def get_model(self, weight_path):
        from src.model.SADRNv2 import get_model
        model = get_model()
        pretrained = torch.load(weight_path, map_location=config.DEVICE)
        model_dict = model.state_dict()
        match_dict = {k: v for k, v in pretrained.items() if (k in model_dict and v.shape == model_dict[k].shape)}
        model_dict.update(match_dict)
        model.load_state_dict(model_dict)
        model = model.to(config.DEVICE)
        model.eval()
        return model
