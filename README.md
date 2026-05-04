# Federated Learning on MNIST

> From-scratch FedAvg in PyTorch · distributed Streamlit simulator · real Substra integration.
> FedAvg implémenté de zéro · simulateur distribué Streamlit · intégration Substra.

🇬🇧 [English](#english) · 🇫🇷 [Français](#français)

Built as a 3-sprint academic project (8INF887 — Apprentissage profond, UQAC).

---

## English

### Highlights

- **From-scratch FedAvg** in PyTorch — see `utils.py` (`federated_average`).
- **Three architectures** — `MLP`, `CNN_V1`, `CNN_V3` (with the BatchNorm /
  non-IID interaction analysed in Sprint 2).
- **Three data distributions** — IID, non-IID by class, imbalanced.
- **Streamlit distributed simulator** — multiple instances acting as Client / Server,
  exchanging models via a shared `models/` directory.
- **Substra integration** — real `substrafl` 1.0.0 pipeline (subprocess backend,
  no Docker required) reusing the same model architectures.

### Results (Sprint 2)

| Model  | IID    | Non-IID classes | Non-IID imbalanced |
|--------|--------|-----------------|--------------------|
| MLP    | ~0.95  | 0.7328          | 0.9720             |
| CNN_V1 | ~0.987 | ~0.85           | ~0.986             |
| CNN_V2 | ~0.985 | **~0.11**       | ~0.988             |
| CNN_V3 | ~0.985 | ~0.83           | ~0.989             |

CNN_V2 collapses to chance on non-IID classes because BatchNorm running statistics
are incompatible across heterogeneous clients — CNN_V3 (BN removed, dropout kept)
restores robustness. Details in [`docs/sprint2_report.pdf`](docs/sprint2_report.pdf).

### Quickstart

```bash
git clone <repo>
cd FL
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The Streamlit app has two tabs:

- **🛰️ Distribué (Client / Server)** — pick a mode in the sidebar. To simulate
  multiple machines, run several instances on different ports:
  ```bash
  streamlit run app.py --server.port 8501   # one tab as Server
  streamlit run app.py --server.port 8502   # one tab as Client 1
  streamlit run app.py --server.port 8503   # one tab as Client 2
  ```
  All instances share `models/`; clients write `client_*.pt`, the server aggregates
  into `global.pt`. Includes a multi-round auto-runner with live convergence chart.

- **🔬 Substra** — runs a real `substrafl` FedAvg compute plan locally (subprocess
  backend, no Docker). MLP / 2 clients / 2 rounds takes ~20s.

Notebook walkthrough:

```bash
jupyter notebook notebooks/federated_mnist.ipynb
```

### Project layout

```
app.py               # Streamlit UI (Distribué + Substra tabs)
fl_models.py         # PyTorch architectures (no I/O side effects)
utils.py             # MNIST loading, splits, training helpers, FedAvg
substra_runner.py    # substrafl pipeline (subprocess backend)
notebooks/           # Jupyter walkthrough + earlier visual demo
docs/                # Sprint reports, architecture diagram, figures
```

### Reports

- [Sprint 1](docs/sprint1_report.pdf) — from-scratch FedAvg (94.88% federated vs
  97.0% centralized, 3 rounds, IID, MLP).
- [Sprint 2](docs/sprint2_report.pdf) — non-IID analysis, CNN architectures,
  autoencoder experiment.

(Reports written in French.)

### Known caveats

- `substrafl` 1.0.0 + PyTorch ≥ 2.6 has a known bug at
  `substrafl/algorithms/pytorch/torch_base_algo.py:249` (`torch.load(path)` needs
  `weights_only=False`). Patch the file in your venv if you hit
  `_pickle.UnpicklingError: Weights only load failed`.
- `utils.py:split_noniid_imbalanced` uses a hardcoded 5-entry proportions list, so
  it implicitly assumes `num_clients = 5` — keep that in mind in the sidebar.

---

## Français

### Aperçu

- **FedAvg implémenté de zéro** en PyTorch — voir `utils.py` (`federated_average`).
- **Trois architectures** — `MLP`, `CNN_V1`, `CNN_V3` (avec l'interaction
  BatchNorm / non-IID analysée dans le Sprint 2).
- **Trois distributions de données** — IID, non-IID par classes, déséquilibré.
- **Simulateur distribué Streamlit** — plusieurs instances en mode Client / Server
  qui échangent leurs modèles via le dossier partagé `models/`.
- **Intégration Substra** — vrai pipeline `substrafl` 1.0.0 (backend subprocess,
  sans Docker) qui réutilise les mêmes architectures.

### Résultats (Sprint 2)

| Modèle | IID    | Non-IID classes | Non-IID déséquilibré |
|--------|--------|-----------------|----------------------|
| MLP    | ~0.95  | 0.7328          | 0.9720               |
| CNN_V1 | ~0.987 | ~0.85           | ~0.986               |
| CNN_V2 | ~0.985 | **~0.11**       | ~0.988               |
| CNN_V3 | ~0.985 | ~0.83           | ~0.989               |

CNN_V2 chute au niveau du hasard en non-IID par classes : ses statistiques de
BatchNorm sont incompatibles entre clients hétérogènes. CNN_V3 (BN retiré, Dropout
conservé) restaure la robustesse. Détails dans
[`docs/sprint2_report.pdf`](docs/sprint2_report.pdf).

### Démarrage rapide

```bash
git clone <repo>
cd FL
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

L'app Streamlit a deux onglets :

- **🛰️ Distribué (Client / Server)** — choisis un mode dans la sidebar. Pour
  simuler plusieurs machines, lance plusieurs instances sur des ports différents :
  ```bash
  streamlit run app.py --server.port 8501   # un onglet en Server
  streamlit run app.py --server.port 8502   # un onglet en Client 1
  streamlit run app.py --server.port 8503   # un onglet en Client 2
  ```
  Toutes les instances partagent `models/` ; les clients écrivent `client_*.pt`,
  le serveur agrège dans `global.pt`. Inclut un auto-runner multi-rounds avec
  courbe de convergence en direct.

- **🔬 Substra** — lance un vrai compute plan `substrafl` FedAvg en local
  (backend subprocess, sans Docker). MLP / 2 clients / 2 rounds prend ~20 s.

Notebook pédagogique :

```bash
jupyter notebook notebooks/federated_mnist.ipynb
```

### Structure du projet

```
app.py               # UI Streamlit (onglets Distribué + Substra)
fl_models.py         # Architectures PyTorch (sans side-effects à l'import)
utils.py             # Chargement MNIST, splits, helpers d'entraînement, FedAvg
substra_runner.py    # Pipeline substrafl (backend subprocess)
notebooks/           # Notebook + démo visuelle Sprint 2
docs/                # Rapports de sprint, diagramme d'archi, figures
```

### Rapports

- [Sprint 1](docs/sprint1_report.pdf) — FedAvg from-scratch (94.88 % fédéré vs
  97.0 % centralisé, 3 rounds, IID, MLP).
- [Sprint 2](docs/sprint2_report.pdf) — analyse non-IID, architectures CNN,
  expérimentation avec un autoencodeur.
- [Rendu](docs/rapport_final.pdf) — Rapport Final.
