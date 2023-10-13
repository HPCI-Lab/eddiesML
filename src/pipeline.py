import matplotlib.pyplot as plt
import numpy as np
import os
import random
import sys
import torch
import torch_geometric
from torch_geometric.loader import DataLoader
from torch_geometric.nn import summary
import xarray as xr
import yaml

import Dataset
import Models
import Loss
from utils import time_func

print(f"Torch version: {torch.__version__}")
print(f"Cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Cuda device: {torch.cuda.get_device_name()}")
print(f"Cuda version: {torch.version.cuda}")
print(f"Torch geometric version: {torch_geometric.__version__}")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)

yaml_file = sys.argv[1]
params = yaml.safe_load(open(yaml_file))

DATA_PATH = params['input_subset_pre_processed']
MESH_PATH = params['input_subset_grid']

TRAIN_PROP = params['train_prop']
VAL_PROP = params['val_prop']
TEST_PROP = params['test_prop']
TRAIN_VAL_TEST = [TRAIN_PROP, VAL_PROP, TEST_PROP]

TRAIN_BATCH_SIZE = params['train_batch_size']
VAL_BATCH_SIZE = params['val_batch_size']
TEST_BATCH_SIZE = params['test_batch_size']

N_FEATURES = params['n_features']
HID_CHANNELS = params['hid_channels']
N_CLASSES = params['n_classes']

FINAL_ACT = None
if params['final_act'] == "sigmoid":
    FINAL_ACT = torch.sigmoid
elif params['final_act'] == "softmax":
    FINAL_ACT = torch.softmax
elif params['final_act'] == "linear":
    FINAL_ACT = torch.nn.Linear(1, 1)

LOSS_OP = None
if params['loss_op'] == "CE":
    LOSS_OP = torch.nn.CrossEntropyLoss()
elif params['loss_op'] == "WCE":
    class_weights = [params['loss_weight_1'], params['loss_weight_2'], params['loss_weight_3']]
    LOSS_OP = Loss.WeightedCrossEntropyLoss(class_weights, DEVICE)

OPTIMIZER = None
if params['optimizer'] == "Adam":
    OPTIMIZER = torch.optim.Adam

LEARN_RATE = params['learn_rate']

EPOCHS = params['epochs']

PLOT_SHOW = params['plot_show']
PLOT_FOLDER = params['output_images_path']
PLOT_N = params['plot_number']

TIMESTAMP = time_func.start_time()


random_seed = random.randint(1, 10000)
print(f"Random seed for train-val-test split: {random_seed}")

timestamp = time_func.start_time()

train_dataset = Dataset.EddyDataset(root=DATA_PATH, mesh_path=MESH_PATH, split='train', proportions=TRAIN_VAL_TEST, random_seed=random_seed)
val_dataset = Dataset.EddyDataset(root=DATA_PATH, mesh_path=MESH_PATH, split='val', proportions=TRAIN_VAL_TEST, random_seed=random_seed)
test_dataset = Dataset.EddyDataset(root=DATA_PATH, mesh_path=MESH_PATH, split='test', proportions=TRAIN_VAL_TEST, random_seed=random_seed)

time_func.stop_time(timestamp, "Datasets creation")


print(train_dataset.len(), val_dataset.len(), test_dataset.len())


if (TRAIN_PROP+VAL_PROP+TEST_PROP) != 100:
    raise ValueError(f"Sum of train-val-test proportions with value {TRAIN_PROP+VAL_PROP+TEST_PROP} is different from 1")

if FINAL_ACT == None:
    raise ValueError(f"Parameter 'final_act' is invalid with value {params['final_act']}")

if LOSS_OP == None:
    raise ValueError(f"Parameter 'loss_op' is invalid with value {params['loss_op']}")

if OPTIMIZER == None:
    raise ValueError(f"Parameter 'optimizer' is invalid with value {params['optimizer']}")

dummy_graph = train_dataset[0]

if dummy_graph.num_features != N_FEATURES:
    raise ValueError(f"Graph num_features is different from parameter N_FEATURES: ({dummy_graph.num_features} != {N_FEATURES})")

if dummy_graph.is_directed():
    raise ValueError("Graph edges are directed!")


train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=6, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False, num_workers=6, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=6, pin_memory=True)

print(len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))


Model = Models.GUNet

model = Model(
    in_channels = N_FEATURES,
    hidden_channels = HID_CHANNELS,
    out_channels = N_CLASSES,
    num_nodes = dummy_graph.num_nodes,   # TODO can put these in Dataset.py
    final_act = FINAL_ACT
).to(DEVICE)

print(model)


dummy_graph.to(DEVICE)
print(summary(model, dummy_graph))


OPTIMIZER = OPTIMIZER(model.parameters(), lr=LEARN_RATE)


def train():
    model.train()
    total_loss = 0

    for batch in train_loader:
        batch = batch.to(DEVICE)

        # zero the parameter gradients
        OPTIMIZER.zero_grad()

        # forward + loss
        pred = model(batch)
        loss = LOSS_OP(pred, batch.y)

        # If you try the Soft Dice Score, use this(even if the loss stays constant)
        #loss.requires_grad = True
        #loss = torch.tensor(loss.item(), requires_grad=True)

        total_loss += loss.item() * batch.num_graphs
        
        # backward + optimize
        loss.backward()
        OPTIMIZER.step()

    average_loss = total_loss / len(train_loader.dataset)
    return average_loss


@torch.no_grad()
def evaluate(loader):
    model.eval()
    total_loss = 0

    for batch in loader:
        batch = batch.to(DEVICE)

        # forward + loss
        pred = model(batch)
        loss = LOSS_OP(pred, batch.y)

        total_loss += loss.item() * batch.num_graphs
    
    average_loss = total_loss / len(loader.dataset)
    return average_loss


time_func.stop_time(TIMESTAMP, "Computation before training finished!")


timestamp = time_func.start_time()

train_loss = []
valid_loss = []

for epoch in range(EPOCHS):
    t_loss = train()
    v_loss = evaluate(val_loader)
    print(f'Epoch: {epoch+1:03d}, Train running loss: {t_loss:.4f}, Val running loss: {v_loss:.4f}')
    train_loss.append(t_loss)
    valid_loss.append(v_loss)

time_func.stop_time(timestamp, "Training Complete!")

metric = evaluate(test_loader)
print(f'Metric for test: {metric:.4f}')


plt.figure(figsize=(8, 4))
plt.plot(train_loss, label='Train loss')
plt.plot(valid_loss, label='Validation loss')
plt.legend(title="Loss type: " + params['loss_op'])

if PLOT_SHOW:
    plt.show()
else:
    plt.savefig(PLOT_FOLDER+"/train_val_losses_" + str(PLOT_N) + ".png")
    plt.close()


torch.no_grad()
model.eval()
batch = next(iter(test_loader))
batch = batch.to(DEVICE)
pred = model(batch)
print(pred)


mesh = xr.open_dataset(MESH_PATH)
mesh_lon = mesh.lon[mesh.nodes].values
mesh_lat = mesh.lat[mesh.nodes].values


batch.to('cpu')
this_target = batch.y[:mesh.dims['nodes_subset']]
this_pred = []
for p in pred[:mesh.dims['nodes_subset']]:
    p = p.tolist()
    max_value = max(p)
    max_index = p.index(max_value)
    this_pred.append(max_index)


fig, axes = plt.subplots(2, 1, figsize=(12, 12))

im = axes[0].scatter(mesh_lon, mesh_lat, c=this_target, s=1)
im2 = axes[1].scatter(mesh_lon, mesh_lat, c=this_pred, s=1)

if PLOT_SHOW:
    plt.show()
else:
    plt.savefig(PLOT_FOLDER+"/pred_vs_ground_" + str(PLOT_N) + ".png")
    plt.close()


timestamp = time_func.start_time()

torch.no_grad()
model.eval()
correct_pred = 0
tot_pred = 0
tot_background = 0

for batch in test_loader:
    batch = batch.to(DEVICE)
    pred = model(batch)
    tot_pred += len(pred)
    
    pred_values = []
    for p in pred:
        p = p.tolist()
        max_value = max(p)
        max_index = p.index(max_value)
        pred_values.append(max_index)
    
    for b in batch.y:
        if b==0:
            tot_background += 1
    
    if len(pred_values) != len(batch.y):
        raise ValueError("Just to be extra sure, but you should never see this error appear.")
    
    for i in range(len(batch.y)):
        if pred_values[i] == batch.y[i]:
            correct_pred += 1

print(f"Total background cells:\t{tot_background}")
print(f"Correct predictions:\t{correct_pred}")
print(f"Total predictions:\t{tot_pred}")
print(f"Graph U-Net accuracy:\t{correct_pred/tot_pred:.4f}")

time_func.stop_time(timestamp, "Accuracy calculated!\n")