"""Chargement MNIST, fonctions de split, helpers d'entraînement 
et implémentation FedAvg (utilitaires)."""


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
import numpy as np

from fl_models import SimpleMLP, CNN_V1, CNN_V3, get_model  # re-exported for compat

# =========================
# CONFIG
# =========================

BATCH_SIZE = 64
NUM_WORKERS = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =========================
# DATASET
# =========================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# =========================
# SPLITS
# =========================

def split_dataset_iid(dataset, num_clients):
    total_len = len(dataset)
    base = total_len // num_clients
    lengths = [base] * num_clients
    for i in range(total_len - base*num_clients):
        lengths[i] += 1
    return random_split(dataset, lengths)


def split_noniid_classes(dataset, num_clients):
    targets = np.array(dataset.targets)
    class_splits = np.array_split(np.arange(10), num_clients)

    subsets = []
    for classes in class_splits:
        idx = np.where(np.isin(targets, classes))[0]
        subsets.append(Subset(dataset, idx))

    return subsets


def split_noniid_imbalanced(dataset, num_clients):
    total_len = len(dataset)
    proportions = [0.6, 0.2, 0.1, 0.05, 0.05]

    indices = np.random.permutation(total_len)
    splits = (np.array(proportions) * total_len).astype(int)

    subsets = []
    start = 0
    for s in splits:
        subsets.append(Subset(dataset, indices[start:start+s]))
        start += s

    return subsets


def get_data_split(split_type, dataset, num_clients):
    if split_type == "iid":
        return split_dataset_iid(dataset, num_clients)
    elif split_type == "noniid_classes":
        return split_noniid_classes(dataset, num_clients)
    elif split_type == "imbalanced":
        return split_noniid_imbalanced(dataset, num_clients)
    else:
        raise ValueError("Unknown split")


# =========================
# TRAINING
# =========================

def train_one_epoch(model, dataloader, optimizer):
    model.train()
    criterion = nn.CrossEntropyLoss()

    for xb, yb in dataloader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()


def evaluate(model):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

    return correct / total


# =========================
# FEDERATED
# =========================

def federated_average(state_dicts, weights):
    total = float(sum(weights))
    norm_weights = [w / total for w in weights]

    avg_state = {}

    for key in state_dicts[0].keys():
        if state_dicts[0][key].dtype.is_floating_point:
            avg_state[key] = torch.zeros_like(state_dicts[0][key])
            for sd, w in zip(state_dicts, norm_weights):
                avg_state[key] += sd[key] * w
        else:
            avg_state[key] = state_dicts[0][key]

    return avg_state


def run_federated_simulation(model_name, split_type, num_clients, rounds, lr):

    global_model = get_model(model_name).to(device)
    global_state = global_model.state_dict()

    subsets = get_data_split(split_type, train_dataset, num_clients)

    loaders = [
        DataLoader(s, batch_size=BATCH_SIZE, shuffle=True)
        for s in subsets
    ]

    acc_per_round = []

    for r in range(rounds):
        local_states = []
        local_sizes = []

        for loader in loaders:
            local_model = get_model(model_name).to(device)
            local_model.load_state_dict(global_state)

            optimizer = torch.optim.SGD(local_model.parameters(), lr=lr)

            train_one_epoch(local_model, loader, optimizer)

            local_states.append(local_model.state_dict())
            local_sizes.append(len(loader.dataset))

        global_state = federated_average(local_states, local_sizes)
        global_model.load_state_dict(global_state)

        acc = evaluate(global_model)
        acc_per_round.append(acc)

    return acc_per_round