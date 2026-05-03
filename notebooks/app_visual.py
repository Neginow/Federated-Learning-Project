import streamlit as st
import matplotlib.pyplot as plt
import time

from utils import run_federated_simulation

st.set_page_config(page_title="FL Simulator", layout="wide")

# =========================
# STYLE
# =========================

st.markdown("""
    <style>
    .big-title {
        font-size:32px !important;
        font-weight:700;
    }
    .card {
        padding:15px;
        border-radius:10px;
        background-color:#111;
        margin-bottom:10px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-title">Federated Learning Simulator</p>', unsafe_allow_html=True)

# =========================
# LAYOUT
# =========================

left, right = st.columns([1, 2])

# =========================
# LEFT PANEL
# =========================

with left:
    st.markdown("### ⚙️ Configuration")

    model_name = st.selectbox("Model", ["MLP", "CNN_V1", "CNN_V3"])
    split_type = st.selectbox("Data", ["iid", "noniid_classes", "imbalanced"])

    num_clients = st.slider("Clients", 2, 10, 5)
    rounds = st.slider("Rounds", 1, 10, 5)
    lr = st.slider("Learning Rate", 0.01, 0.2, 0.1)

    run = st.button("🚀 Run")

# =========================
# RIGHT PANEL
# =========================
with right:

    if run:

        st.markdown("## 📊 Dashboard")

        # placeholders
        metric_placeholder = st.empty()
        graph_placeholder = st.empty()
        flow_placeholder = st.empty()

        results = []

        for r in range(rounds):

            # =========================
            # 🟡 STEP 1 — CLIENT TRAINING
            # =========================
            with flow_placeholder.container():
                st.markdown(f"### 🔄 Round {r+1}")

                cols = st.columns(num_clients)
                for i, c in enumerate(cols):
                    c.markdown(f"🟡 Client {i+1}\n\nTraining...")

            time.sleep(0.5)

            # =========================
            # 🔵 STEP 2 — SENDING
            # =========================
            with flow_placeholder.container():
                cols = st.columns(num_clients)
                for i, c in enumerate(cols):
                    c.markdown(f"🔵 Client {i+1}\n\nSending →")

            time.sleep(0.5)

            # =========================
            # 🧠 STEP 3 — SERVER
            # =========================
            with flow_placeholder.container():
                st.markdown("### 🧠 Server")
                st.markdown("Aggregating models...")

            time.sleep(0.5)

            # =========================
            # ⚙️ ACTUAL TRAINING (backend)
            # =========================
            partial = run_federated_simulation(
                model_name=model_name,
                split_type=split_type,
                num_clients=num_clients,
                rounds=1,
                lr=lr
            )

            if len(results) == 0:
                results = partial
            else:
                results.append(partial[-1])

            # =========================
            # 📦 STEP 4 — GLOBAL MODEL
            # =========================
            with flow_placeholder.container():
                st.markdown("### 📦 Global Model Updated")
                st.markdown(f"Accuracy: **{results[-1]:.4f}**")

            time.sleep(0.5)

            # =========================
            # METRIC
            # =========================
            metric_placeholder.metric(
                label=f"Accuracy (Round {r+1})",
                value=f"{results[-1]:.4f}"
            )

            # =========================
            # GRAPH LIVE
            # =========================
            fig, ax = plt.subplots()
            ax.plot(range(1, len(results)+1), results, linewidth=3, marker='o')
            ax.set_title("Model Convergence")
            ax.set_xlabel("Rounds")
            ax.set_ylabel("Accuracy")
            ax.grid(alpha=0.3)

            graph_placeholder.pyplot(fig)

        st.success("Training complete ✅")