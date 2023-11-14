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

DATASET_SIZE = params['dataset_size']

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

train_dataset = Dataset.EddyDataset(root=DATA_PATH, mesh_path=MESH_PATH, dataset_size=DATASET_SIZE, split='train', proportions=TRAIN_VAL_TEST, random_seed=random_seed)
val_dataset = Dataset.EddyDataset(root=DATA_PATH, mesh_path=MESH_PATH, dataset_size=DATASET_SIZE, split='val', proportions=TRAIN_VAL_TEST, random_seed=random_seed)
test_dataset = Dataset.EddyDataset(root=DATA_PATH, mesh_path=MESH_PATH, dataset_size=DATASET_SIZE, split='test', proportions=TRAIN_VAL_TEST, random_seed=random_seed)

time_func.stop_time(timestamp, "Datasets creation")


print(train_dataset.len(), val_dataset.len(), test_dataset.len())


if (TRAIN_PROP+VAL_PROP+TEST_PROP) != 100:
    raise ValueError(f"Sum of train-val-test proportions with value {TRAIN_PROP+VAL_PROP+TEST_PROP} is different from 100")

if FINAL_ACT == None:
    raise ValueError(f"Parameter 'final_act' is invalid with value {params['final_act']}")

if LOSS_OP == None:
    if params['loss_op'] != "Dice":
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


if params['loss_op'] == "Dice":
    
    timestamp = time_func.start_time()

    tot_counts = [0, 0, 0]
    for batch in train_loader:
        batch = batch.to(DEVICE)
        
        unique, counts = torch.unique(batch.y, return_counts=True)
        
        # TODO - I don't really like this, it just informs me whether something is wrong and then does it anyway
        if 0 not in unique:
            raise ValueError("Error: class 0 not present in batch")
        elif 1 not in unique:
            raise ValueError("Error: class 1 not present in batch")
        elif 2 not in unique:
            raise ValueError("Error: class 2 not present in batch")
        else:
            for class_idx in unique:
                tot_counts[class_idx] += counts[class_idx].item()

    time_func.stop_time(timestamp, "Unique counted!")
    
    freq = [c/np.sum(tot_counts) for c in tot_counts]
    freq_inv = [1/f for f in freq]
    class_weights = [f/np.sum(freq_inv) for f in freq_inv]
    print(freq_inv, "- freq_inv")
    print(class_weights, "- class_weights")
    LOSS_OP = Loss.SoftDiceLoss(class_weights)


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


timestamp = time_func.start_time()
DEVICE=torch.device('cpu')
model = model.to(DEVICE)


model.eval()
with torch.no_grad():
    batch = next(iter(test_loader))
    batch = batch.to(DEVICE)
    pred = model(batch)


mesh = xr.open_dataset(MESH_PATH)
mesh_lon = mesh.lon[mesh.nodes].values
mesh_lat = mesh.lat[mesh.nodes].values


this_target = batch.y[:mesh.dims['nodes_subset']]
_, this_pred = torch.max(pred[:mesh.dims['nodes_subset']], dim=1)


fig, axes = plt.subplots(2, 1, figsize=(12, 12))

im = axes[0].scatter(mesh_lon, mesh_lat, c=this_target, s=1)
im2 = axes[1].scatter(mesh_lon, mesh_lat, c=this_pred, s=1)

if PLOT_SHOW:
    plt.show()
else:
    plt.savefig(PLOT_FOLDER+"/pred_vs_ground_" + str(PLOT_N) + ".png")
    plt.close()

time_func.stop_time(timestamp, "pred_vs_ground plot created!")


# Running it on cuda is a huge improvement
DEVICE=torch.device('cuda')
model = model.to(DEVICE)

timestamp = time_func.start_time()

model.eval()
with torch.no_grad():
    tot_background = 0
    correct_pred = 0
    tot_pred = len(test_loader.dataset)*dummy_graph.num_nodes

    for batch in test_loader:
        batch = batch.to(DEVICE)

        pred = model(batch)

        _, indices = torch.max(pred, dim=1)

        tot_background += (batch.y == 0).sum().item()

        # This works because the values in the indices correspond to the values in batch.y
        correct_pred += (indices == batch.y).sum().item()

    print(f"Total background cells:\t{tot_background}")
    print(f"Correct predictions:\t{correct_pred}")
    print(f"Total predictions:\t{tot_pred}")
    print(f"Graph U-Net accuracy:\t{correct_pred/tot_pred*100:.2f}%")

time_func.stop_time(timestamp, "Accuracy calculated!")
