import sys, os

from os.path import join

# from src.configs.config_SADRN_v2 import *
from src.configs.config_SADRN_v2_eval import *
# from src.configs.config_PPRN_pretrain1 import *
from src.configs.config_AFLW2000 import *

INIT_IMAEG_SIZE = 450
CROPPED_IMAGE_SIZE = 256
UV_MAP_SIZE = 256

base_path = os.path.dirname(__file__)

UV_MEAN_SHAPE_PATH    = join(base_path, 'data/uv_data/mean_shape_map.npy')
UV_FACE_MASK_PATH     = join(base_path, 'data/uv_data/uv_face_mask.png')
BFM_UV_MAT_PATH       = join(base_path, 'data/Out/BFM_UV.mat')
UV_KPT_INDEX_PATH     = join(base_path, 'data/uv_data/uv_kpt_ind.txt')
FACE_WEIGHT_MASK_PATH = join(base_path, 'data/uv_data/uv_weight_mask.png')
UV_EDGES_PATH         = join(base_path, 'data/uv_data/uv_edges.npy')
UV_TRIANGLES_PATH     = join(base_path, 'data/uv_data/uv_triangles.npy')

DEVICE = 'cuda:0'
