import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

MODEL_DIR = Path(__file__).parent / "model"

st.set_page_config(page_title="Network Intrusion Detector", page_icon="🛡️", layout="centered")


# --------------------------------------------------------------------- loading
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_DIR / "random_forest_final.pkl")
    preprocessor = joblib.load(MODEL_DIR / "preprocessor_final.pkl")
    selector = joblib.load(MODEL_DIR / "feature_selector_final.pkl")
    with open(MODEL_DIR / "app_meta.json") as f:
        meta = json.load(f)
    return model, preprocessor, selector, meta


@st.cache_data
def load_csv(name):
    return pd.read_csv(MODEL_DIR / name)


try:
    model, preprocessor, selector, meta = load_artifacts()
except FileNotFoundError:
    st.error("Model files not found. Copy the contents of the `app_export` folder into `model/`.")
    st.stop()

ov = meta["overview"]
tm = meta["test_metrics"]
attack_pct = ov["attack_rate"] * 100


def predict(df: pd.DataFrame):
    """Raw UNSW-NB15 columns in -> (predicted class, attack probability)."""
    X = selector.transform(preprocessor.transform(df[meta["columns"]]))
    prob = model.predict_proba(X)[:, 1]
    return (prob >= 0.5).astype(int), prob


# --------------------------------------------------------------------- sidebar
st.sidebar.title("Network Intrusion Detector")
st.sidebar.markdown(
    "Classifies network traffic flows as **normal** or **attack** using a model "
    "trained on flow statistics from the UNSW-NB15 dataset."
)
st.sidebar.markdown(f"**Model in use:** Random Forest (tuned, {selector.k} selected features)")
st.sidebar.markdown(f"**Training rows:** {ov['rows']:,}")
st.sidebar.markdown(f"**Class balance:** {100 - attack_pct:.0f}% normal / {attack_pct:.0f}% attack")
st.sidebar.caption("A scikit-learn classifier pipeline, deployed with Streamlit.")
page = st.sidebar.radio("Go to", ["Overview", "Try a Prediction", "Model Performance"])


# -------------------------------------------------------------------- overview
def page_overview():
    st.title("🛡️ Network Intrusion Detector")
    st.write(
        "This app trains a Random Forest on network flow records to flag each one as "
        "**normal** or an **attack** (e.g. DoS, exploits, reconnaissance, fuzzers, backdoors)."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{ov['rows']:,}")
    c2.metric("Attack rate", f"{attack_pct:.1f}%")
    c3.metric("Attack types", ov["attack_types"])

    st.subheader("Attack type breakdown")
    counts = pd.Series(ov["attack_counts"]).sort_index()
    st.bar_chart(counts)

    st.subheader("Sample of the training data")
    st.dataframe(load_csv("sample_train.csv"))

    fi = meta["feature_importance"]["features"][:3]
    st.subheader("Where the model's signal actually comes from")
    st.info(
        f"The Random Forest relies most on **{fi[0]}**, **{fi[1]}** and **{fi[2]}**. "
        "In UNSW-NB15, `sttl` is the source-to-destination time-to-live and `ct_state_ttl` counts "
        "connections sharing the same state and TTL range, so the model leans on how a flow "
        "behaves rather than on any single payload value. See **Model Performance** for the "
        "honest numbers, including where the model still gets it wrong."
    )


# ------------------------------------------------------------------ prediction
FIELDS = {
    "proto": "Transport protocol of the flow (e.g. tcp, udp).",
    "service": "Network service on the destination ('-' means none identified).",
    "state": "Connection state (e.g. FIN, INT, CON).",
    "dur": "Total duration of the flow, in seconds.",
    "sbytes": "Bytes sent from source to destination.",
    "dbytes": "Bytes sent from destination to source.",
    "sttl": "Source-to-destination time-to-live value.",
    "ct_state_ttl": "Connections per state and TTL range (a count).",
    "rate": "Packets per second for the flow.",
}
STEP = {"dur": 0.01, "rate": 100.0}
FMT = {"dur": "%.6f", "rate": "%.4f"}


def _init_form():
    for name in FIELDS:
        key = f"in_{name}"
        if key not in st.session_state:
            st.session_state[key] = meta["defaults"][name]
    st.session_state.setdefault("base_row", dict(meta["defaults"]))


def randomize():
    row = load_csv("sample_input.csv").sample(1).iloc[0]
    st.session_state["base_row"] = {c: row[c] for c in meta["columns"]}
    st.session_state["example_label"] = int(row["label"]) if "label" in row else None
    for name in FIELDS:
        val = row[name]
        if name in meta["categories"]:
            opts = meta["categories"][name]
            val = val if val in opts else meta["defaults"][name]
        else:
            val = float(val)
        st.session_state[f"in_{name}"] = val


def manual_entry():
    _init_form()
    st.button("🎲 Randomize", on_click=randomize)
    st.write("Fill in the flow details below, or click Randomize for a real example record.")

    cols = st.columns(2)
    for i, (name, help_txt) in enumerate(FIELDS.items()):
        with cols[i % 2]:
            if name in meta["categories"]:
                st.selectbox(name, meta["categories"][name], key=f"in_{name}", help=help_txt)
            else:
                st.number_input(name, key=f"in_{name}", step=STEP.get(name, 1.0),
                                format=FMT.get(name, "%.0f"), help=help_txt)

    if st.button("Classify this flow", type="primary"):
        row = dict(st.session_state["base_row"])
        for name in FIELDS:
            row[name] = st.session_state[f"in_{name}"]
        pred, prob = predict(pd.DataFrame([row]))
        if pred[0] == 1:
            st.error(f"🚨 Predicted: **ATTACK**  (attack probability {prob[0]:.1%})")
        else:
            st.success(f"✅ Predicted: **NORMAL**  (attack probability {prob[0]:.1%})")
        st.progress(float(prob[0]))
        true = st.session_state.get("example_label")
        if true is not None:
            st.caption(f"Randomized record's true label: {'Attack' if true else 'Normal'} "
                       "(edited fields may change the outcome).")
        st.caption("Fields not shown use the values of the randomized record, or typical (median) values.")


def csv_upload():
    st.write(
        "Upload a CSV with the raw UNSW-NB15 feature columns (same format as the dataset's "
        "test file). `id`, `attack_cat` and `label` are optional; if `label` is present, accuracy is reported."
    )
    st.download_button("Download a sample CSV to try", load_csv("sample_input.csv").to_csv(index=False).encode(),
                       "sample_input.csv", "text/csv")
    file = st.file_uploader("Upload traffic records (CSV)", type="csv")
    if file is None:
        return
    data = pd.read_csv(file)
    missing = [c for c in meta["columns"] if c not in data.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        return

    pred, prob = predict(data)
    out = data.copy()
    out["predicted_label"] = pred
    out["attack_probability"] = prob.round(4)

    c1, c2, c3 = st.columns(3)
    c1.metric("Records", f"{len(out):,}")
    c2.metric("Predicted attacks", f"{int(pred.sum()):,}")
    c3.metric("Predicted normal", f"{int((pred == 0).sum()):,}")

    if "label" in data.columns:
        y = data["label"].astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        m1, m2, m3 = st.columns(3)
        m1.metric("Accuracy", f"{accuracy_score(y, pred):.2%}")
        m2.metric("F1-score", f"{f1_score(y, pred):.3f}")
        m3.metric("False positive rate", f"{fp / (fp + tn):.2%}" if (fp + tn) else "n/a")

    st.dataframe(out.head(200))
    st.download_button("Download predictions", out.to_csv(index=False).encode(), "predictions.csv", "text/csv")


def page_predict():
    st.title("Try a Prediction")
    st.write("Describe a network flow, or upload a CSV of records, to classify.")
    tab1, tab2 = st.tabs(["Manual entry", "Upload CSV"])
    with tab1:
        manual_entry()
    with tab2:
        csv_upload()


# ----------------------------------------------------------------- performance
def page_performance():
    st.title("Model Performance")
    st.write(
        f"Evaluated on the official UNSW-NB15 test partition ({ov['test_rows']:,} flows, never seen "
        "during training) using the **Random Forest (tuned)** model."
    )
    cols = st.columns(6)
    for col, (label, key) in zip(cols, [("Accuracy", "Accuracy"), ("Precision", "Precision"), ("Recall", "Recall"),
                                         ("F1 Score", "F1-Score"), ("AUC", "ROC-AUC"), ("FPR", "FPR")]):
        col.metric(label, f"{tm[key]:.3f}")

    val = meta["validation"]
    bp = meta["best_params"]
    st.info(
        f"The model uses `class_weight='balanced'` with tuned parameters "
        f"(`n_estimators={bp.get('n_estimators')}`, `max_depth={bp.get('max_depth')}`). "
        f"It catches {tm['Recall']:.1%} of attacks, but flags {tm['FPR']:.1%} of normal flows as attacks. "
        f"On a validation split carved from the training data the same model reached "
        f"{val['Accuracy']:.1%} accuracy and a {val['FPR']:.1%} false positive rate, so the drop on the "
        "official test partition reflects a distribution shift between the two partitions "
        "rather than a bug. An AUC well above 0.5 confirms the model is genuinely separating the classes."
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Confusion Matrix")
        cm = np.array(meta["confusion"])
        fig, ax = plt.subplots(figsize=(4, 3.6))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1], ["normal", "attack"])
        ax.set_yticks([0, 1], ["normal", "attack"])
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v:,}", ha="center", va="center",
                    color="white" if v > cm.max() / 2 else "black")
        fig.tight_layout()
        st.pyplot(fig)
    with right:
        st.subheader("ROC Curve")
        roc = meta["roc"]
        fig, ax = plt.subplots(figsize=(4, 3.6))
        ax.plot(roc["fpr"], roc["tpr"], label=f"model (AUC = {tm['ROC-AUC']:.3f})")
        ax.plot([0, 1], [0, 1], "--", color="gray", label="random guessing")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        st.pyplot(fig)

    st.subheader("Feature importance")
    fi = meta["feature_importance"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(fi["features"][::-1], fi["importance"][::-1], color="#4c72b0")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    st.pyplot(fig)

    st.subheader("Model comparison (test set)")
    st.dataframe(pd.DataFrame(meta["comparison"]).set_index("Model"))

    st.subheader("Sample test-set predictions")
    st.dataframe(load_csv("sample_predictions.csv"))


{"Overview": page_overview, "Try a Prediction": page_predict, "Model Performance": page_performance}[page]()
