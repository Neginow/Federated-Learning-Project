import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets, transforms

HERE = Path(__file__).resolve().parent
OPENER_PATH = HERE / "opener.py"
FL_MODELS_PATH = HERE / "fl_models.py"
SUBSTRA_RUNNER_PATH = HERE / "substra_runner.py"
DATASET_DESCRIPTION_PATH = HERE / "dataset_description.md"

from substra import Client, BackendType
from substra.sdk.schemas import DatasetSpec, DataSampleSpec, Permissions

from substrafl.algorithms.pytorch import TorchFedAvgAlgo
from substrafl.dependency import Dependency
from substrafl.evaluation_strategy import EvaluationStrategy
from substrafl.experiment import execute_experiment
from substrafl.index_generator import NpIndexGenerator
from substrafl.nodes import AggregationNode, TestDataNode, TrainDataNode
from substrafl.strategies import FedAvg

from fl_models import get_model


# =========================
# DATASET
# =========================

class TorchDataset(Dataset):
    def __init__(self, data_from_opener, is_inference=False):
        self.x = torch.tensor(data_from_opener["images"], dtype=torch.float32)
        self.y = torch.tensor(data_from_opener["labels"], dtype=torch.int64)
        self.is_inference = is_inference

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if self.is_inference:
            return self.x[idx]
        return self.x[idx], self.y[idx]


def accuracy(data_from_opener, predictions):
    labels = np.asarray(data_from_opener["labels"])
    preds = np.asarray(predictions)

    if preds.ndim == 2:
        preds = preds.argmax(axis=1)

    return float(np.mean(preds == labels))


# =========================
# DATA SPLIT
# =========================

def prepare_partitions(num_clients, split_type):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_ds = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST("./data", train=False, download=True, transform=transform)

    def to_arrays(ds):
        xs = torch.stack([ds[i][0] for i in range(len(ds))]).numpy()
        ys = np.array([ds[i][1] for i in range(len(ds))])
        return xs, ys

    train_x, train_y = to_arrays(train_ds)
    test_x, test_y = to_arrays(test_ds)

    rng = np.random.default_rng(42)

    if split_type == "iid":
        idx = rng.permutation(len(train_x))
        chunks = np.array_split(idx, num_clients)

    elif split_type == "noniid_classes":
        class_splits = np.array_split(np.arange(10), num_clients)
        chunks = []
        for classes in class_splits:
            mask = np.isin(train_y, classes)
            indices = np.where(mask)[0]
            chunks.append(indices)

    elif split_type == "imbalanced":
        # Same hardcoded ratios as utils.split_noniid_imbalanced (5 entries).
        proportions = np.array([0.6, 0.2, 0.1, 0.05, 0.05])[:num_clients]
        proportions = proportions / proportions.sum()
        idx = rng.permutation(len(train_x))
        sizes = (proportions * len(idx)).astype(int)
        chunks = []
        start = 0
        for s in sizes:
            chunks.append(idx[start:start + s])
            start += s

    else:
        raise ValueError(f"Unknown split type: {split_type}")

    tmp = Path(tempfile.mkdtemp())

    paths = []
    for i, chunk in enumerate(chunks):
        d = tmp / f"org_{i}"
        d.mkdir()

        np.save(d / "images.npy", train_x[chunk])
        np.save(d / "labels.npy", train_y[chunk])

        paths.append(d)

    test_dir = tmp / "test"
    test_dir.mkdir()

    np.save(test_dir / "images.npy", test_x)
    np.save(test_dir / "labels.npy", test_y)

    return paths, test_dir, tmp


# =========================
# ALGO
# =========================

def make_algo(model_name, lr, dataset_size, batch_size, num_updates):

    if num_updates is None:
        num_updates = dataset_size // batch_size  # 1 epoch

    print(f"[INFO] num_updates = {num_updates}")

    class Algo(TorchFedAvgAlgo):
        def __init__(self):
            model = get_model(model_name)

            super().__init__(
                model=model,
                criterion=nn.CrossEntropyLoss(),
                optimizer=torch.optim.SGD(model.parameters(), lr=lr),
                index_generator=NpIndexGenerator(
                    num_updates=num_updates,
                    batch_size=batch_size
                ),
                dataset=TorchDataset,
                seed=42
            )

    return Algo


# =========================
# MAIN FUNCTION
# =========================


def run_substra_experiment(
    model_name="CNN_V3",
    num_clients=3,
    num_rounds=5,
    lr=0.1,
    num_updates=None,
    batch_size=64,
    progress_callback=None,
    split_type="noniid_classes"
):

    # =========================
    # PROGRESS HELPER
    # =========================
    def report(stage, **info):
        if progress_callback:
            progress_callback(stage, info)

    print("\n===== SUBSTRA RUN =====")

    report("preparing_data", num_clients=num_clients, split_type=split_type)

    paths, test_path, tmp_dir = prepare_partitions(num_clients, split_type)

    dataset_size = len(np.load(paths[0] / "labels.npy"))

    report("creating_clients", n=num_clients)

    clients = [
        Client(client_name=f"org_{i}", backend_type=BackendType.LOCAL_SUBPROCESS)
        for i in range(num_clients)
    ]

    org_ids = [c.organization_info().organization_id for c in clients]

    permissions = Permissions(public=False, authorized_ids=org_ids)

    dataset_keys = []
    sample_keys = []

    report("registering_data")

    for i, client in enumerate(clients):
        ds = client.add_dataset(
            DatasetSpec(
                name=f"MNIST {i}",
                data_opener=str(OPENER_PATH),
                description=str(DATASET_DESCRIPTION_PATH),
                permissions=permissions,
                logs_permission=permissions,
            )
        )

        ss = client.add_data_sample(
            DataSampleSpec(
                path=str(paths[i]),
                data_manager_keys=[ds],
            )
        )

        dataset_keys.append(ds)
        sample_keys.append(ss)

    # Register a dedicated test dataset/sample on org 0 so accuracy is measured
    # on the held-out MNIST test set, not on org 0's training partition.
    test_dataset_key = clients[0].add_dataset(
        DatasetSpec(
            name="MNIST test",
            data_opener=str(OPENER_PATH),
            description=str(DATASET_DESCRIPTION_PATH),
            permissions=permissions,
            logs_permission=permissions,
        )
    )
    test_sample_key = clients[0].add_data_sample(
        DataSampleSpec(
            path=str(test_path),
            data_manager_keys=[test_dataset_key],
        )
    )

    train_nodes = [
        TrainDataNode(
            organization_id=org_ids[i],
            data_manager_key=dataset_keys[i],
            data_sample_keys=[sample_keys[i]],
        )
        for i in range(num_clients)
    ]

    test_node = TestDataNode(
        organization_id=org_ids[0],
        data_manager_key=test_dataset_key,
        data_sample_keys=[test_sample_key],
    )

    Algo = make_algo(model_name, lr, dataset_size, batch_size, num_updates)

    strategy = FedAvg(
        algo=Algo(),
        metric_functions={"accuracy": accuracy}
    )

    report("running", rounds=num_rounds)

    compute_plan = execute_experiment(
        client=clients[0],
        strategy=strategy,
        train_data_nodes=train_nodes,
        evaluation_strategy=EvaluationStrategy([test_node], eval_frequency=1),
        aggregation_node=AggregationNode(organization_id=org_ids[0]),
        num_rounds=num_rounds,
        experiment_folder="tmp_exp",
        dependencies=Dependency(
            pypi_dependencies=["torch", "torchvision", "numpy"],
            # Both files are needed in the subprocess: fl_models.py supplies
            # `get_model`, and substra_runner.py is the module the Algo class
            # lives in (cloudpickle resolves it by qualified name on unpickle).
            local_code=[FL_MODELS_PATH, SUBSTRA_RUNNER_PATH],
        ),
    )

    report("collecting_results")

    perfs = clients[0].get_performances(compute_plan.key)

    accs = [p for _, p in sorted(zip(perfs.round_idx, perfs.performance))]

    print("Accuracy history:", accs)

    report("complete", final_acc=accs[-1] if accs else None)

    shutil.rmtree(tmp_dir, ignore_errors=True)

    return {
        "acc_history": accs,
        "final_acc": accs[-1] if accs else None,
        "compute_plan_key": str(compute_plan.key),
    }