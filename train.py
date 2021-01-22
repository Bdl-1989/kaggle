import sys
sys.path.append(os.getcwd() + '/src/')
import pandas as pd
import seaborn as sns
from config import *
import os

OUTPUT_DIR = './'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

#TRAIN_PATH = '../input/cassava-leaf-disease-classification/train_images'
TRAIN_PATH = '../input/cassava-leaf-disease-merged/train'
TEST_PATH = '../input/cassava-leaf-disease-classification/test_images'


os.listdir('../input/cassava-leaf-disease-classification')
train = pd.read_csv('../input/cassava-leaf-disease-classification/train.csv')
train_merged = pd.read_csv('../input/cassava-leaf-disease-merged/merged.csv')
test = pd.read_csv('../input/cassava-leaf-disease-classification/sample_submission.csv')
label_map = pd.read_json('../input/cassava-leaf-disease-classification/label_num_to_disease_map.json', 
                         orient='index')
display(label_map)

if CFG.debug:
    CFG.epochs = 1
    train = train.sample(n=1000, random_state=CFG.seed).reset_index(drop=True)

if CFG.device == 'TPU':
    import os
    os.system('curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py')
    os.system('python pytorch-xla-env-setup.py --version nightly --apt-packages libomp5 libopenblas-dev')
    os.system('export XLA_USE_BF16=1')
    os.system('export XLA_TENSOR_ALLOCATOR_MAXSIZE=100000000')
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.distributed.xla_multiprocessing as xmp
    CFG.lr = CFG.lr * CFG.nprocs
    CFG.batch_size = CFG.batch_size // CFG.nprocs

if CFG.rand_augment:
    !pip install git+https://github.com/ildoonet/pytorch-randaugment > /dev/null
    from torchvision.transforms import transforms
    from RandAugment import RandAugment