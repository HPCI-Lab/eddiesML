# %%
# Imports + Global settings
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch_geometric
from torch_geometric.loader import DataLoader

import Dataset
import Models
from utils import time_func

print(f"Torch version: {torch.__version__}")
print(f"Cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Cuda device: {torch.cuda.get_device_name()}")
print(f"Torch geometric version: {torch_geometric.__version__}")

### GLOBAL SETTINGS ###

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_PATH = '../data'

ON_CLUSTER = False
TRAIN_SET = [1982, 1983]
VALID_SET = [1980]
TEST_SET = [1981]
_num_features = None
_hidden_channels = 32
_num_classes = 1
_train_batch_size = 512
_test_batch_size = 512
_valid_batch_size = 512
FINAL_ACTIVATION = torch.sigmoid#torch.nn.Linear(1, 1) for regression tasks
GRAPH_DICT = {}
PLOT_VERTICAL = True

TIMESTAMP = time_func.start_time()


# %%
# Dataset creation
timestamp = time_func.start_time()
dataset = Dataset.PilotDataset(root=DATA_PATH)
time_func.stop_time(timestamp, "Dataset creation")


# %%
# Data loader set up + Feature normalization
# The Dataloader allows to feed data by batch into the model to exploit parallelism
# The "batch" information is used by the GNN model to tell which nodes belong to which graph
# "batch_size" specifies how many grids will be saved in a single batch
# TODO: If "shuffle=True", the data will be reshuffled at every epoch. understand
# if this means that you may potentially train over the same patch/batch over and over
#dataset = dataset.shuffle()
# TODO normalization may not be needed, as we'll only work with the SSH feature


# %%
# Model instantiation and summary
Model = Models.GUNet

model = Model(
    in_channels = _num_features,
    hidden_channels = _hidden_channels,
    out_channels = _num_classes,
    num_nodes = train_loader.dataset[0].num_nodes,  # just initialization
    final_act = FINAL_ACTIVATION
).to(DEVICE)


# %%
# Loss + Optimizer + train()

# Classification losses
loss_op = torch.nn.BCELoss()
#loss_op = torch.nn.CrossEntropyLoss()
#loss_op = torch.nn.BCEWithLogitsLoss()
#loss_op = torch.nn.functional.nll_loss
#from dice import dice_score
#loss_op = dice_score

# Regression losses
#loss_op = torch.nn.L1Loss()
#loss_op = torch.nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

def train():
    model.train()
    total_loss = 0

    for batch in train_loader:
        batch = batch.to(DEVICE)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + loss
        pred = model(batch)
        pred = pred.squeeze()

        loss = loss_op(pred, batch.y)

        # If you try the Soft Dice Score, use this(even if the loss stays constant)
        #loss.requires_grad = True
        #loss = torch.tensor(loss.item(), requires_grad=True)

        # backward + optimize
        # loss * _train_batch_size(5)
        total_loss += loss.item() * batch.num_graphs
        loss.backward()
        optimizer.step()

    # average loss = total_loss / training graps(20)
    total_loss = total_loss / len(train_loader.dataset)
    return total_loss

#loss = train()
#print("Train loss, debug: ", loss)

# %%
# Evaluate()
# torch.no_grad() as decorator is useful because I'm validating/testing, so not calling 
# the backward(). Mouse on it and read the doc
@torch.no_grad()
def evaluate(loader):
    model.eval()
    total_loss = 0

    for batch in loader:
        batch = batch.to(DEVICE)

        pred = model(batch)
        pred = pred.squeeze()

        loss = loss_op(pred, batch.y)

        total_loss += loss.item() * batch.num_graphs    # TODO compare this with the AI at scale code
    
    total_loss = total_loss / len(loader.dataset)
    return total_loss

'''
loss = evaluate(train_loader)
print("Train loss, debug: ", loss)
loss = evaluate(valid_loader)
print("Valid loss, debug: ", loss)
loss = evaluate(test_loader)
print("Test loss, debug: ", loss)
'''

time_func.stop_time(TIMESTAMP, "Computation before training finished!")

# %%
# Settings recap
print(f"Final act: {FINAL_ACTIVATION}")
print(f"Loss function: {loss_op}")

# %%
# Epoch training, validation and testing
timestamp = time_func.start_time()

train_loss = []
valid_loss = []

for epoch in range(100):
    t_loss = train()
    v_loss = evaluate(valid_loader)
    print(f'Epoch: {epoch+1:03d}, Train running loss: {t_loss:.4f}, Val running loss: {v_loss:.4f}')
    train_loss.append(t_loss)
    valid_loss.append(v_loss)

time_func.stop_time(timestamp, "Training Complete!")

metric = evaluate(test_loader)
print(f'Metric for test: {metric:.4f}')

# %%
# Plot train and validation losses
plt.figure(figsize=(8, 4))
plt.plot(train_loss, label='Train loss')
plt.plot(valid_loss, label='Validation loss')
plt.legend(title="Loss type: " + str(loss_op))

if ON_CLUSTER:
    plt.savefig('./images/train_valid_losses.png')
    plt.close()
else:
    plt.show()

# %% 
# Visualization of test batch truth against predictions
@torch.no_grad()
def visual():

    # These default variables are for the horizontal plot
    ncols = 5
    nrows = 2
    figsize = (16, 6)
    col_target = 0
    row_target = 0
    col_pred = 0
    row_pred = 1
    hspace = 0.1
    wspace = 0.5

    if PLOT_VERTICAL:
        ncols = 2
        nrows = 5
        figsize = (6, 14)
        col_target = 0
        row_target = 0
        col_pred = 1
        row_pred = 0
        hspace = 0.05
        wspace = 0.4

    model.eval()

    batch = next(iter(test_loader))
    batch = batch.to(DEVICE)

    pred = model(batch)  # shape: [1600*_test_batch_size, 1]
    pred = pred.squeeze()
    
    # Preparing the plot
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize)

    # Take 4 patches from the batch
    shift = 20
    for patch_id in range(0+shift, 5+shift):#len(test_loader.dataset)):

        # Allocate the empty prediction and target matrices
        mat_pred = np.zeros(shape=(40, 40))
        mat_target = np.zeros(shape=(40, 40))

        # Put the data inside
        index = 0+1600*patch_id
        for lon in range(40):
            for lat in range(40):
                mat_pred[lat, lon] = pred[index].item()
                mat_target[lat, lon] = batch.y[index].item()
                index += 1

        ax_target = axs[row_target, col_target].matshow(mat_target)
        ax_pred = axs[row_pred, col_pred].matshow(mat_pred)
        fig.colorbar(ax_pred, fraction=0.046, pad=0.04)#, format='%.0e')
        fig.colorbar(ax_target, fraction=0.046, pad=0.04)
        
        if PLOT_VERTICAL:
            row_target += 1
            row_pred += 1
        else:
            col_target += 1
            col_pred += 1
    
    plt.subplots_adjust(hspace=hspace, wspace=wspace)

    if ON_CLUSTER:
        plt.savefig('./images/testbatch.png')
        plt.close(fig)
    else:
        plt.show()
    
    #print(test_set[0].y, '\t', np.shape(test_set[0].y))
    print("Pred:\t", pred[:], '\n\t', np.shape(pred[:]))
    print("Target:\t", batch.y[:], '\n\t', np.shape(batch.y[:]))

visual()
