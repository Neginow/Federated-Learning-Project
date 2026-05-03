"""
Substra/substrafl FedAvg pipeline on MNIST, reusing models from fl_models.py.

Backend: substra LOCAL_SUBPROCESS (no Docker needed).
Public entry: run_substra_experiment(...).
"""

import shutil
import tempfile
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets, transforms

# Workaround for substrafl 1.0.0 + PyTorch >=2.6: substrafl calls torch.load(path)
# without weights_only=False, which rejects NpIndexGenerator etc. (see the
# `# TO CHANGE` comment in substrafl/algorithms/pytorch/torch_base_algo.py).
# The subprocess imports this module to unpickle TorchDataset, so this patch
# takes effect before substrafl loads any checkpoint.
_orig_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

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

OPENER_CODE = """\
import os
import numpy as np
import substratools as tools


class MnistOpener(tools.Opener):
    def get_data(self, folders):
        xs, ys = [], []
        for folder in folders:
            xs.append(np.load(os.path.join(folder, "images.npy")))
            ys.append(np.load(os.path.join(folder, "labels.npy")))
        return {
            "images": np.concatenate(xs, axis=0),
            "labels": np.concatenate(ys, axis=0),
        }

    def fake_data(self, n_samples):
        return {
            "images": np.zeros((n_samples, 1, 28, 28), dtype=np.float32),
            "labels": np.zeros(n_samples, dtype=np.int64),
        }
"""

DESCRIPTION_MD = "# MNIST FL\nFederated MNIST partition.\n"


class TorchDataset(Dataset):
    """Wraps the dict returned by MnistOpener.get_data into a torch Dataset.

    Substrafl requires the param to be named `data_from_opener`.
    """

    def __init__(self, data_from_opener, is_inference: bool = False):
        self.x = torch.tensor(data_from_opener["images"], dtype=torch.float32)
        self.y = torch.tensor(data_from_opener["labels"], dtype=torch.int64)
        self.is_inference = is_inference

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if self.is_inference:
            return self.x[idx]
        return self.x[idx], self.y[idx]


def _accuracy(data_from_opener, predictions):
    """Metric callback for substrafl."""
    labels = np.asarray(data_from_opener["labels"])
    preds = np.asarray(predictions)
    if preds.ndim == 2:
        preds = preds.argmax(axis=1)
    return float(np.mean(preds == labels))


def _make_algo_class(model_name: str, lr: float, num_updates: int, batch_size: int):
    """Build a TorchFedAvgAlgo subclass bound to (model_name, lr, num_updates, batch_size)."""

    class _Algo(TorchFedAvgAlgo):
        def __init__(self):
            model = get_model(model_name)
            super().__init__(
                model=model,
                criterion=nn.CrossEntropyLoss(),
                optimizer=torch.optim.SGD(model.parameters(), lr=lr),
                index_generator=NpIndexGenerator(
                    num_updates=num_updates, batch_size=batch_size
                ),
                dataset=TorchDataset,
                seed=42,
            )

    return _Algo


def _prepare_partitions(num_clients: int, data_root: Path):
    """Load MNIST, split IID across clients, write per-client + test as .npy files."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    def to_arrays(ds):
        xs = torch.stack([ds[i][0] for i in range(len(ds))]).numpy().astype(np.float32)
        ys = np.array([ds[i][1] for i in range(len(ds))], dtype=np.int64)
        return xs, ys

    train_x, train_y = to_arrays(train_ds)
    test_x, test_y = to_arrays(test_ds)

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(train_x))
    chunks = np.array_split(idx, num_clients)

    train_paths = []
    for i, chunk in enumerate(chunks):
        d = data_root / f"org_{i}"
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "images.npy", train_x[chunk])
        np.save(d / "labels.npy", train_y[chunk])
        train_paths.append(d)

    test_dir = data_root / "test"
    test_dir.mkdir(parents=True, exist_ok=True)
    np.save(test_dir / "images.npy", test_x)
    np.save(test_dir / "labels.npy", test_y)

    return train_paths, test_dir


def run_substra_experiment(
    model_name: str = "MLP",
    num_clients: int = 3,
    num_rounds: int = 2,
    lr: float = 0.1,
    num_updates: int = 50,
    batch_size: int = 64,
    progress_callback: Optional[Callable[[str, dict], None]] = None,
) -> dict:
    """Run a Substra/substrafl FedAvg experiment on MNIST.

    Returns a dict with keys:
        acc_history       : list[float], per-round test accuracy
        final_acc         : float
        num_rounds        : int
        compute_plan_key  : str
    """

    def report(stage: str, **info):
        if progress_callback:
            progress_callback(stage, info)

    workspace = Path(tempfile.mkdtemp(prefix="substra_fl_"))
    data_root = workspace / "data"
    opener_path = workspace / "opener.py"
    description_path = workspace / "description.md"
    experiment_folder = workspace / "experiment"
    experiment_folder.mkdir()

    opener_path.write_text(OPENER_CODE)
    description_path.write_text(DESCRIPTION_MD)

    try:
        report("preparing_data", num_clients=num_clients)
        train_paths, test_path = _prepare_partitions(num_clients, data_root)

        report("creating_clients", n=num_clients)
        clients = [
            Client(client_name=f"org_{i}", backend_type=BackendType.LOCAL_SUBPROCESS)
            for i in range(num_clients)
        ]
        org_ids = [c.organization_info().organization_id for c in clients]
        algo_org_id = org_ids[0]
        permissions = Permissions(public=False, authorized_ids=org_ids)

        report("registering_data")
        dataset_keys, sample_keys = [], []
        for i, client in enumerate(clients):
            ds_key = client.add_dataset(
                DatasetSpec(
                    name=f"MNIST org {i}",
                    data_opener=str(opener_path),
                    description=str(description_path),
                    permissions=permissions,
                    logs_permission=permissions,
                )
            )
            ss_key = client.add_data_sample(
                DataSampleSpec(
                    path=str(train_paths[i]),
                    data_manager_keys=[ds_key],
                )
            )
            dataset_keys.append(ds_key)
            sample_keys.append(ss_key)

        test_ds_key = clients[0].add_dataset(
            DatasetSpec(
                name="MNIST test",
                data_opener=str(opener_path),
                description=str(description_path),
                permissions=permissions,
                logs_permission=permissions,
            )
        )
        test_ss_key = clients[0].add_data_sample(
            DataSampleSpec(path=str(test_path), data_manager_keys=[test_ds_key])
        )

        train_nodes = [
            TrainDataNode(
                organization_id=org_ids[i],
                data_manager_key=dataset_keys[i],
                data_sample_keys=[sample_keys[i]],
            )
            for i in range(num_clients)
        ]
        test_nodes = [
            TestDataNode(
                organization_id=org_ids[0],
                data_manager_key=test_ds_key,
                data_sample_keys=[test_ss_key],
            )
        ]
        aggregation_node = AggregationNode(organization_id=algo_org_id)

        AlgoCls = _make_algo_class(model_name, lr, num_updates, batch_size)
        strategy = FedAvg(algo=AlgoCls(), metric_functions={"accuracy": _accuracy})

        eval_strategy = EvaluationStrategy(
            test_data_nodes=test_nodes, eval_frequency=1
        )

        deps = Dependency(
            pypi_dependencies=["torch", "torchvision", "numpy"],
            local_code=[
                Path(__file__).resolve(),
                Path(__file__).resolve().parent / "fl_models.py",
            ],
        )

        report("running", num_rounds=num_rounds)
        compute_plan = execute_experiment(
            client=clients[0],
            strategy=strategy,
            train_data_nodes=train_nodes,
            evaluation_strategy=eval_strategy,
            aggregation_node=aggregation_node,
            num_rounds=num_rounds,
            experiment_folder=str(experiment_folder),
            dependencies=deps,
        )

        report("collecting_results")
        perfs = clients[0].get_performances(compute_plan.key)
        # parallel arrays — zip round_idx with performance, sort by round_idx
        rounds_and_acc = sorted(zip(perfs.round_idx, perfs.performance))
        acc_history = [a for _, a in rounds_and_acc]
        final_acc = acc_history[-1] if acc_history else None

        report("complete", final_acc=final_acc)

        return {
            "acc_history": acc_history,
            "final_acc": final_acc,
            "num_rounds": num_rounds,
            "compute_plan_key": str(compute_plan.key),
        }
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    # Smoke test
    def cb(stage, info):
        print(f"[{stage}] {info}")

    result = run_substra_experiment(
        model_name="MLP",
        num_clients=2,
        num_rounds=2,
        lr=0.1,
        num_updates=30,
        batch_size=64,
        progress_callback=cb,
    )
    print("RESULT:", result)
