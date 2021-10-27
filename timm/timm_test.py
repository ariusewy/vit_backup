from models import create_model
import torch
import torch.nn as nn

v = create_model("vit_small_patch16_224", pretrained=True)
v.head = nn.Linear(v.head.in_features, 10)
print('Building ViT Model:\n{}'.format(v))
print(v.state_dict)