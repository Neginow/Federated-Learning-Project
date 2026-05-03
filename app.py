"""
Federated Learning — distributed simulator + Substra integration.

Two top-level tabs:
- Distribué : Client/Server multi-instance protocol (file-based) with
  manual aggregation, multi-round auto-runner, convergence chart.
- Substra  : Real substrafl 1.0.0 FedAvg pipeline (subprocess backend).

Sidebar config applies to the Distribué tab only. Multi-instance launch:

    streamlit run app.py --server.port 8501   # Server
    streamlit run app.py --server.port 8502   # Client 1
    streamlit run app.py --server.port 8503   # Client 2
    ...
"""

import glob
import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from matplotlib.patches import Patch
from torch.utils.data import DataLoader

from utils import (
    BATCH_SIZE,
    device,
    evaluate,
    federated_average,
    get_data_split,
    get_model,
    train_dataset,
)

MODELS_DIR = "models"
GLOBAL_PATH = os.path.join(MODELS_DIR, "global.pt")
SEED = 42

os.makedirs(MODELS_DIR, exist_ok=True)

st.set_page_config(page_title="FL Distributed", layout="wide")


# =========================
# Helpers
# =========================

def list_client_files():
    return sorted(glob.glob(os.path.join(MODELS_DIR, "client_*.pt")))


def fmt_time(ts):
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def deterministic_subsets(split_type, num_clients):
    # Same seed on every instance -> client {id} sees the same subset everywhere.
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    return get_data_split(split_type, train_dataset, num_clients)


def train_local(model, loader, lr, progress_bar):
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    n_batches = len(loader)
    running_loss = 0.0
    for i, (xb, yb) in enumerate(loader):
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        progress_bar.progress((i + 1) / n_batches)
    return running_loss / max(n_batches, 1)


def show_client_view(subset, split_type):
    """Affiche ce que voit le client : message, classes présentes, histogramme."""
    targets = np.array(subset.dataset.targets)
    labels = targets[list(subset.indices)]
    counts = np.bincount(labels, minlength=10)
    classes_present = sorted({int(c) for c in labels})

    if split_type == "noniid_classes":
        st.warning("⚠️ Ce client ne voit qu'un sous-ensemble des classes → scénario non-IID")
    elif split_type == "imbalanced":
        st.warning("⚠️ Distribution déséquilibrée des données")
    else:
        st.success("✔️ Distribution homogène des données (IID)")

    c1, c2 = st.columns(2)
    c1.metric("Échantillons locaux", int(len(labels)))
    c2.metric("Classes présentes", f"{len(classes_present)} / 10")
    st.write(f"**Classes présentes :** {set(classes_present)}")

    fig, ax = plt.subplots(figsize=(6, 3))
    colors = ["#4C9AFF" if counts[i] > 0 else "#dddddd" for i in range(10)]
    ax.bar(range(10), counts, color=colors, edgecolor="#222")
    ax.set_xticks(range(10))
    ax.set_xlabel("Classe")
    ax.set_ylabel("Nombre d'échantillons")
    ax.set_title("Distribution locale par classe")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(
        handles=[
            Patch(facecolor="#4C9AFF", edgecolor="#222", label="Présent"),
            Patch(facecolor="#dddddd", edgecolor="#222", label="Absent"),
        ],
        loc="upper right",
        fontsize=8,
    )
    st.pyplot(fig)
    plt.close(fig)


def plot_acc_history(history, title="Convergence du modèle global"):
    fig, ax = plt.subplots(figsize=(6, 3))
    rounds = list(range(1, len(history) + 1))
    ax.plot(rounds, history, marker="o", linewidth=2, color="#4C9AFF")
    if 0 < len(rounds) <= 20:
        ax.set_xticks(rounds)
    ax.set_xlabel("Round")
    ax.set_ylabel("Global accuracy")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.0)
    return fig


def run_auto_rounds(n_rounds, lr, model_name, split_type, num_clients):
    """Simule N rounds FL en-process via le protocole disque. Retourne acc_history."""
    if not os.path.exists(GLOBAL_PATH):
        st.error("Initialise d'abord le modèle global.")
        return None

    init_blob = torch.load(GLOBAL_PATH, map_location=device, weights_only=False)
    if init_blob.get("model_name") != model_name:
        st.error(
            f"Le modèle global est `{init_blob.get('model_name')}`, "
            f"mais l'UI sélectionne `{model_name}`."
        )
        return None

    subsets = deterministic_subsets(split_type, num_clients)
    acc_history = list(init_blob.get("acc_history", []))
    base_round = int(init_blob.get("round", 0))

    progress = st.progress(0.0)
    status = st.empty()
    chart_slot = st.empty()
    metric_slot = st.empty()

    total_steps = n_rounds * num_clients
    step = 0
    criterion = nn.CrossEntropyLoss()
    final_acc = None

    for r in range(n_rounds):
        round_num = base_round + r + 1

        global_blob = torch.load(GLOBAL_PATH, map_location=device, weights_only=False)
        global_state = global_blob["state_dict"]

        for cid in range(1, num_clients + 1):
            status.write(f"Round {round_num} — entraînement client {cid}/{num_clients}…")
            model = get_model(model_name).to(device)
            model.load_state_dict(global_state)
            subset = subsets[cid - 1]
            loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True)
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)
            model.train()
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                criterion(model(xb), yb).backward()
                optimizer.step()
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "num_samples": len(subset),
                    "client_id": cid,
                    "model_name": model_name,
                    "timestamp": time.time(),
                    "round": round_num,
                },
                os.path.join(MODELS_DIR, f"client_{cid}.pt"),
            )
            step += 1
            progress.progress(step / total_steps)

        status.write(f"Round {round_num} — agrégation…")
        state_dicts, weights = [], []
        for f in list_client_files():
            cb = torch.load(f, map_location=device, weights_only=False)
            state_dicts.append(cb["state_dict"])
            weights.append(cb["num_samples"])
        new_state = federated_average(state_dicts, weights)

        eval_model = get_model(model_name).to(device)
        eval_model.load_state_dict(new_state)
        final_acc = evaluate(eval_model)
        acc_history.append(final_acc)

        torch.save(
            {
                "state_dict": new_state,
                "model_name": model_name,
                "timestamp": time.time(),
                "round": round_num,
                "acc_history": acc_history,
            },
            GLOBAL_PATH,
        )

        fig = plot_acc_history(acc_history)
        chart_slot.pyplot(fig)
        plt.close(fig)
        metric_slot.metric(f"Round {round_num}", f"{final_acc:.4f}")

    status.success(
        f"✅ Terminé — {n_rounds} rounds, accuracy finale = {final_acc:.4f}"
    )
    return acc_history


# =========================
# Sidebar — applies to Distribué tab
# =========================

st.sidebar.markdown("## Mode (onglet Distribué)")
mode = st.sidebar.selectbox("Run as", ["Client", "Server"])

st.sidebar.markdown("## Shared config")
st.sidebar.caption("Doit être identique sur toutes les instances Distribué.")
model_name = st.sidebar.selectbox("Model", ["MLP", "CNN_V1", "CNN_V3"])
split_type = st.sidebar.selectbox("Data split", ["iid", "noniid_classes", "imbalanced"])
num_clients = st.sidebar.slider("Total clients", 2, 10, 5)


# =========================
# Top-level tabs
# =========================

tab_dist, tab_substra = st.tabs(["🛰️ Distribué (Client/Server)", "🔬 Substra"])


# =========================
# DISTRIBUÉ TAB
# =========================

with tab_dist:
    if mode == "Client":
        st.title("🧑‍💻 Client")

        client_id = st.number_input(
            "Client ID", min_value=1, max_value=num_clients, value=1, step=1
        )
        lr = st.slider("Learning rate", 0.01, 0.2, 0.1)

        if os.path.exists(GLOBAL_PATH):
            gblob = torch.load(GLOBAL_PATH, map_location=device, weights_only=False)
            st.info(f"📡 Round global actuel : **{int(gblob.get('round', 0))}**")
        else:
            st.info("📡 Aucun modèle global encore — attends l'init du serveur.")

        subsets = deterministic_subsets(split_type, num_clients)
        subset = subsets[int(client_id) - 1]
        show_client_view(subset, split_type)

        st.divider()

        if st.button("🚀 Train local model"):
            if not os.path.exists(GLOBAL_PATH):
                st.error("`models/global.pt` introuvable — initialise-le côté serveur.")
                st.stop()

            blob = torch.load(GLOBAL_PATH, map_location=device, weights_only=False)
            if blob.get("model_name") != model_name:
                st.error(
                    f"Le modèle global est `{blob.get('model_name')}`, "
                    f"mais l'UI sélectionne `{model_name}`. Aligne la config."
                )
                st.stop()

            model = get_model(model_name).to(device)
            model.load_state_dict(blob["state_dict"])

            loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True)
            progress = st.progress(0.0)
            with st.spinner("Entraînement (1 epoch)…"):
                avg_loss = train_local(model, loader, lr, progress)

            with st.spinner("Évaluation sur le test set…"):
                acc = evaluate(model)

            client_path = os.path.join(MODELS_DIR, f"client_{int(client_id)}.pt")
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "num_samples": len(subset),
                    "client_id": int(client_id),
                    "model_name": model_name,
                    "timestamp": time.time(),
                    "round": int(blob.get("round", 0)),
                },
                client_path,
            )

            st.success(f"Sauvegardé → `{client_path}`")
            c1, c2 = st.columns(2)
            c1.metric("Loss locale (moy.)", f"{avg_loss:.4f}")
            c2.metric("Test accuracy", f"{acc:.4f}")

    else:
        st.title("🧠 Server")

        if os.path.exists(GLOBAL_PATH):
            _gb = torch.load(GLOBAL_PATH, map_location=device, weights_only=False)
            st.info(
                f"📡 Round actuel : **{int(_gb.get('round', 0))}** "
                f"· {len(_gb.get('acc_history', []))} round(s) historisé(s)"
            )

        c1, c2 = st.columns(2)

        with c1:
            if st.button("⚡ Initialize global model"):
                torch.manual_seed(SEED)
                model = get_model(model_name).to(device)
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "model_name": model_name,
                        "timestamp": time.time(),
                        "round": 0,
                        "acc_history": [],
                    },
                    GLOBAL_PATH,
                )
                for f in list_client_files():
                    os.remove(f)
                st.success(f"Initialisé `{GLOBAL_PATH}` (round=0, anciens clients supprimés).")

        with c2:
            if st.button("🔗 Aggregate models"):
                files = list_client_files()
                if not files:
                    st.error("Aucun modèle client trouvé dans `models/`.")
                else:
                    state_dicts, weights, skipped = [], [], []
                    for f in files:
                        blob = torch.load(f, map_location=device, weights_only=False)
                        if blob.get("model_name") != model_name:
                            skipped.append(os.path.basename(f))
                            continue
                        state_dicts.append(blob["state_dict"])
                        weights.append(blob["num_samples"])

                    if skipped:
                        st.warning(f"Ignorés (modèle incompatible) : {', '.join(skipped)}")

                    if not state_dicts:
                        st.error("Aucun client compatible à agréger.")
                    else:
                        prev = torch.load(GLOBAL_PATH, map_location=device, weights_only=False)
                        new_round = int(prev.get("round", 0)) + 1
                        history = list(prev.get("acc_history", []))

                        new_state = federated_average(state_dicts, weights)
                        eval_model = get_model(model_name).to(device)
                        eval_model.load_state_dict(new_state)
                        with st.spinner("Évaluation du nouveau modèle global…"):
                            acc = evaluate(eval_model)
                        history.append(acc)

                        torch.save(
                            {
                                "state_dict": new_state,
                                "model_name": model_name,
                                "timestamp": time.time(),
                                "round": new_round,
                                "acc_history": history,
                            },
                            GLOBAL_PATH,
                        )

                        st.success(
                            f"Round {new_round} — agrégation de {len(state_dicts)} clients → `{GLOBAL_PATH}`"
                        )
                        st.metric("Global test accuracy", f"{acc:.4f}")

        st.divider()
        st.markdown("### 🔁 Run FL for N rounds (auto)")
        st.caption(
            "Simule le cycle complet (clients → agrégation → re-broadcast) pendant N rounds. "
            "Les fichiers `client_*.pt` et `global.pt` sont mis à jour à chaque round."
        )
        rc1, rc2, rc3 = st.columns([1, 1, 2])
        with rc1:
            n_rounds = st.slider("Nombre de rounds", 1, 20, 5)
        with rc2:
            auto_lr = st.slider("Learning rate", 0.01, 0.2, 0.1, key="auto_lr")
        with rc3:
            st.write("")
            st.write("")
            run_auto = st.button("▶️ Run FL for N rounds")

        if run_auto:
            run_auto_rounds(n_rounds, auto_lr, model_name, split_type, num_clients)

        if os.path.exists(GLOBAL_PATH):
            history = list(
                torch.load(GLOBAL_PATH, map_location=device, weights_only=False).get(
                    "acc_history", []
                )
            )
            if history:
                st.markdown("### 📈 Convergence (historique)")
                fig = plot_acc_history(history)
                st.pyplot(fig)
                plt.close(fig)

        st.divider()
        st.markdown("### Clients détectés")
        files = list_client_files()
        if not files:
            st.info("Aucun modèle client reçu pour l'instant.")
        else:
            rows = []
            for f in files:
                try:
                    blob = torch.load(f, map_location=device, weights_only=False)
                    rows.append(
                        {
                            "file": os.path.basename(f),
                            "client_id": blob.get("client_id", "?"),
                            "model": blob.get("model_name", "?"),
                            "samples": blob.get("num_samples", "?"),
                            "received": fmt_time(
                                blob.get("timestamp", os.path.getmtime(f))
                            ),
                        }
                    )
                except Exception as e:
                    rows.append({"file": os.path.basename(f), "error": str(e)})
            st.table(rows)

        st.markdown("### Modèle global")
        if os.path.exists(GLOBAL_PATH):
            blob = torch.load(GLOBAL_PATH, map_location=device, weights_only=False)
            st.write(f"**Architecture :** `{blob.get('model_name', '?')}`")
            st.write(
                f"**Dernière mise à jour :** "
                f"{fmt_time(blob.get('timestamp', os.path.getmtime(GLOBAL_PATH)))}"
            )
            if st.button("📊 Evaluate global model"):
                model = get_model(blob["model_name"]).to(device)
                model.load_state_dict(blob["state_dict"])
                with st.spinner("Évaluation…"):
                    acc = evaluate(model)
                st.metric("Global test accuracy", f"{acc:.4f}")
        else:
            st.warning("Pas encore de modèle global — clique *Initialize global model*.")


# =========================
# SUBSTRA TAB
# =========================

with tab_substra:
    st.title("🔬 Substra — Real FedAvg")
    st.caption(
        "Pipeline substrafl 1.0.0, backend subprocess (sans Docker). "
        "Réutilise les architectures depuis `fl_models.py`. "
        "Chaque run enregistre des datasets/data samples par organisation, "
        "exécute un compute plan FedAvg, puis collecte les performances."
    )

    sc1, sc2 = st.columns(2)
    with sc1:
        sub_model = st.selectbox(
            "Model", ["MLP", "CNN_V1", "CNN_V3"], key="sub_model"
        )
        sub_clients = st.slider("Nombre de clients", 2, 5, 3, key="sub_clients")
        sub_rounds = st.slider("Nombre de rounds", 1, 5, 2, key="sub_rounds")
    with sc2:
        sub_lr = st.slider("Learning rate", 0.01, 0.2, 0.1, key="sub_lr")
        sub_updates = st.slider(
            "Updates / round", 10, 200, 50, step=10, key="sub_updates"
        )
        sub_batch = st.select_slider(
            "Batch size", options=[16, 32, 64, 128], value=64, key="sub_batch"
        )

    if st.button("▶️ Run Substra experiment"):
        try:
            from substra_runner import run_substra_experiment
        except ImportError as e:
            st.error(
                f"substrafl manquant : {e}\n\n"
                "Installe avec : `pip install substra substrafl substratools`"
            )
            st.stop()

        STAGES = [
            "preparing_data",
            "creating_clients",
            "registering_data",
            "running",
            "collecting_results",
            "complete",
        ]
        status_slot = st.empty()
        progress_bar = st.progress(0.0)

        def cb(stage, info):
            if stage in STAGES:
                progress_bar.progress((STAGES.index(stage) + 1) / len(STAGES))
            extra = " ".join(f"{k}={v}" for k, v in info.items())
            status_slot.info(f"⏳ {stage} {extra}".strip())

        try:
            with st.spinner(
                f"Running Substra: {sub_model}, {sub_clients} clients × {sub_rounds} rounds…"
            ):
                result = run_substra_experiment(
                    model_name=sub_model,
                    num_clients=sub_clients,
                    num_rounds=sub_rounds,
                    lr=sub_lr,
                    num_updates=sub_updates,
                    batch_size=sub_batch,
                    progress_callback=cb,
                )
        except Exception as e:
            status_slot.error(f"Échec : {type(e).__name__}: {e}")
            st.stop()

        status_slot.success(
            f"✅ Done — final accuracy = {result['final_acc']:.4f}"
        )
        st.metric("Accuracy finale", f"{result['final_acc']:.4f}")

        st.markdown("### 📈 Convergence (Substra)")
        fig = plot_acc_history(
            result["acc_history"], title="Convergence Substra (FedAvg)"
        )
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("### Récap rounds")
        st.dataframe(
            [
                {"round": i, "test accuracy": f"{a:.4f}"}
                for i, a in enumerate(result["acc_history"])
            ]
        )

        st.caption(f"Compute plan key : `{result['compute_plan_key']}`")
