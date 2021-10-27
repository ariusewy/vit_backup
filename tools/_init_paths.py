import os.path as osp
import sys
from IPython.core.debugger import Pdb
ipdb = Pdb()

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)
path = '/home/luanzhuo/workplace_wy/ViT_SE'
add_path(path)
