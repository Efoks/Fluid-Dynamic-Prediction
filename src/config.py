import os
import torch

# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = 'cpu'
LR = 1e-3

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.normpath('D:\\EAGLE')

SPLITS_DIR = os.path.join(PROJECT_DIR, 'data', 'Splits')
MODEL_DIR = os.path.join(PROJECT_DIR, 'models')
VIDEOS_DIR = os.path.join(PROJECT_DIR, 'videos')
