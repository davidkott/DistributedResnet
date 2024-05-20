import torch
import torchvision
import torchvision.transforms as transforms
import time
import numpy as np
from torch import nn
import os
import torch._dynamo as TD
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from plotly.offline import plot
from torch.nn.utils import (
  parameters_to_vector as Params2Vec,
  vector_to_parameters as Vec2Params
)
from CustomDataset import CustomDataset
import loss_landscapes
import loss_landscapes.metrics

Othertransform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0, 0, 0], [1, 1, 1])
])
TD.config.verbose=True
model = torchvision.models.resnet50(weights=None)
state = torch.load(r'SavedStates/161.pytorch', map_location='cpu')
state = {f'{k[10:]}': v for k, v in state.items()}
model.load_state_dict(state )
# model = torch.compile(model)

loss_fn = nn.CrossEntropyLoss()
val_dataset = CustomDataset(csv_file= 'images.csv', root_dir = '/s/b/proj/gol_bitvec/val',transform = Othertransform,CutOff = 8_000)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1_000, shuffle=False,pin_memory=True, num_workers=120)
model.eval()
criterion = torch.nn.CrossEntropyLoss()
# contour plot resolutio
STEPS = 40
_,x, y = iter(val_loader).__next__()
metric = loss_landscapes.metrics.Loss(criterion, x, y)
loss_data_fin = loss_landscapes.random_plane(model, metric,distance= .1, steps=STEPS, normalization='filter', deepcopy_model=True)
np.save('Data',loss_data_fin)
fig = plt.figure()
ax = plt.axes(projection='3d')
X = np.array([[j for j in range(STEPS)] for i in range(STEPS)])
Y = np.array([[i for _ in range(STEPS)] for i in range(STEPS)])
ax.plot_surface(X, Y, np.log10(loss_data_fin), rstride=1, cstride=1, cmap='viridis', edgecolor='none')
ax.set_title('Surface Plot of Loss Landscape')
plt.savefig('temp.png')