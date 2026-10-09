import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

# --------------------------------------------------------------- configuration
ACCENT = "#2F6FAF"  # mid blue: readable on both light and dark backgrounds
MODEL_DIR = Path(__file__).parent / "model"
AUTHORS = ""  # e.g. "Juan Dela Cruz, Maria Santos"  (leave empty to hide the byline)

st.set_page_config(page_title="Random Forest Intrusion Detection · UNSW-NB15",
                   page_icon="🛡️", layout="centered", initial_sidebar_state="collapsed")

st.markdown(
    f"""
<style>
[data-testid="stSidebar"], [data-testid="collapsedControl"] {{display:none;}}
.block-container {{max-width: 980px; padding-top: 4rem; padding-bottom: 3rem;}}
/* Text colors are inherited from the active theme (light or dark); only the accent is fixed. */
h1, h2, h3 {{font-family: Georgia, 'Times New Roman', serif; color: {ACCENT}; font-weight: 600;}}
h2 {{margin-top: 1.6rem;}}
.hero {{border-bottom: 3px solid {ACCENT}; padding-bottom: 1.1rem; margin-bottom: 1.2rem;}}
.hero .kicker {{text-transform: uppercase; letter-spacing: .09em; font-size: .74rem; opacity: .65; margin-bottom:.3rem;}}
.hero h1 {{margin: 0; font-size: 2.15rem; line-height: 1.2;}}
.hero .sub {{opacity: .8; font-size: 1.05rem; margin-top: .45rem;}}
.hero .by {{opacity: .65; font-size: .88rem; margin-top: .5rem;}}
div[data-testid="stMetric"] {{background: rgba(127,127,127,.08); border: 1px solid rgba(127,127,127,.25);
    border-left: 4px solid {ACCENT}; border-radius: 6px; padding: .75rem 1rem;}}
div[data-testid="stMetricLabel"] p {{font-size: .82rem; opacity: .75;}}
.callout {{background: rgba(47,111,175,.10); border-left: 4px solid {ACCENT}; padding: .9rem 1.15rem;
    border-radius: 4px; margin: .6rem 0 1rem 0;}}
.callout b.t {{color: {ACCENT};}}
.small {{opacity: .7; font-size: .85rem;}}
button[data-baseweb="tab"] p {{font-size: .95rem; font-weight: 500;}}
footer {{visibility: hidden;}}
</style>
""",
    unsafe_allow_html=True,
)


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

ov, tm, val, bp = meta["overview"], meta["test_metrics"], meta["validation"], meta["best_params"]
cm = np.array(meta["confusion"])
tn, fp, fn, tp = cm.ravel()
train_attack_pct = ov["attack_rate"] * 100
test_attack_pct = (fn + tp) / cm.sum() * 100
n_raw = len(meta["columns"])
n_encoded = len(preprocessor.get_feature_names_out())
k_sel = int(selector.k)
top_feats = meta["feature_importance"]["features"]


def predict(df: pd.DataFrame):
    """Raw UNSW-NB15 columns in -> (predicted class, attack probability)."""
    X = selector.transform(preprocessor.transform(df[meta["columns"]]))
    prob = model.predict_proba(X)[:, 1]
    return (prob >= 0.5).astype(int), prob


def callout(title, body):
    st.markdown(f'<div class="callout"><b class="t">{title}</b><br>{body}</div>', unsafe_allow_html=True)


def finish(fig):
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    return fig


def tidy(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#9aa5b1")
    ax.spines["bottom"].set_color("#9aa5b1")
    ax.tick_params(colors="#3e4c59")


@st.cache_resource
def cm_fig():
    fig, ax = plt.subplots(figsize=(4.2, 3.7))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["Normal", "Attack"])
    ax.set_yticks([0, 1], ["Normal", "Attack"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=11,
                color="white" if v > cm.max() / 2 else "#1f2933")
    return finish(fig)


@st.cache_resource
def roc_fig():
    roc = meta["roc"]
    fig, ax = plt.subplots(figsize=(4.2, 3.7))
    ax.plot(roc["fpr"], roc["tpr"], color=ACCENT, linewidth=2, label=f"Random Forest (AUC = {tm['ROC-AUC']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#9aa5b1", label="Random guessing")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    tidy(ax)
    return finish(fig)


@st.cache_resource
def importance_fig():
    fi = meta["feature_importance"]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.barh(fi["features"][::-1], fi["importance"][::-1], color=ACCENT)
    ax.set_xlabel("Importance (mean decrease in impurity)")
    tidy(ax)
    return finish(fig)


@st.cache_resource
def attack_fig():
    s = pd.Series(ov["attack_counts"]).sort_values()
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.barh(s.index, s.values, color=ACCENT)
    ax.set_xlabel("Records in training partition")
    for y, v in enumerate(s.values):
        ax.text(v, y, f" {v:,}", va="center", fontsize=8, color="#3e4c59")
    ax.set_xlim(0, s.max() * 1.15)
    tidy(ax)
    return finish(fig)


# ---------------------------------------------------------------------- header
by = f'<div class="by">{AUTHORS}</div>' if AUTHORS else ""
st.markdown(
    f"""
<div class="hero">
  <div class="kicker">University of Mindanao · College of Computing Education · CST9 · 1st Semester, S.Y. 2026–2027</div>
  <h1>Random Forest Algorithm for Network Intrusion Detection Using the UNSW-NB15 Dataset</h1>
  <div class="sub">An interactive demonstration of a binary (Normal vs. Attack) network flow classifier.</div>
  {by}
</div>
""",
    unsafe_allow_html=True,
)

tabs = st.tabs(["Overview", "Background", "Dataset", "Try a Prediction", "Performance", "Notes & Limitations"])

# -------------------------------------------------------------------- overview
with tabs[0]:
    c = st.columns(4)
    c[0].metric("Training flows", f"{ov['rows']:,}")
    c[1].metric("Test flows", f"{ov['test_rows']:,}")
    c[2].metric("Features used", f"{k_sel} of {n_encoded}")
    c[3].metric("Test recall", f"{tm['Recall']:.1%}")

    st.header("Project overview")
    st.write(
        "Computer networks are constantly exposed to malicious activity, and intrusion detection systems "
        "(IDS) are expected to separate harmful traffic from legitimate traffic quickly and reliably. "
        "This project trains a **Random Forest** classifier on labelled network flow records from the "
        "UNSW-NB15 dataset and evaluates how well it flags each flow as **normal** or an **attack**. "
        "This web application presents the study, lets you classify flows yourself, and reports the "
        "model's results honestly, including where it falls short."
    )

    st.subheader("Objectives")
    st.markdown(
        f"""
**General objective:** to develop and evaluate a Random Forest model that detects malicious network
traffic in the UNSW-NB15 dataset.

**Specific objectives**
1. Clean and preprocess the UNSW-NB15 training and testing partitions (encoding categorical features and scaling numeric ones).
2. Reduce the {n_encoded} encoded features to the {k_sel} most informative using mutual information, and check how the number of features affects performance.
3. Train and tune a Random Forest, using class weighting to handle the imbalance between normal and attack flows.
4. Evaluate the model on the official test partition using accuracy, precision, recall, F1-score, ROC-AUC and false positive rate.
5. Deploy the trained model as a web application so that it can be explored interactively.
"""
    )

    st.subheader("Scope")
    st.markdown(
        """
- **Task:** binary classification only (Normal = 0, Attack = 1). The nine attack families are not predicted individually.
- **Data:** the official UNSW-NB15 training and testing partitions; no live traffic capture.
- **Algorithm:** Random Forest only.
- **Use:** an academic demonstration, not a production security tool.
"""
    )

    st.subheader("Methodology at a glance")
    steps = [
        ("1 · Load", "Read the two official partitions; confirmed no missing, duplicate or infinite values."),
        ("2 · Clean", "Dropped `id` (row index) and `attack_cat` (would leak the answer)."),
        ("3 · Preprocess", f"One-hot encoded `proto`, `service`, `state`; standardized numeric features ({n_encoded} columns)."),
        ("4 · Select", f"Kept the top {k_sel} features by mutual information with the label."),
        ("5 · Train and tune", f"Random Forest with `class_weight='balanced'`; randomized search on a validation split (best: {bp.get('n_estimators')} trees, max_depth {bp.get('max_depth')})."),
        ("6 · Evaluate", "Scored once on the official test partition and reported all metrics."),
    ]
    cols = st.columns(3)
    for i, (t, d) in enumerate(steps):
        with cols[i % 3]:
            st.markdown(f"**{t}**")
            st.caption(d)

    callout(
        "Key findings",
        f"The model detects <b>{tm['Recall']:.1%}</b> of attacks on the test partition (accuracy {tm['Accuracy']:.1%}, "
        f"F1 {tm['F1-Score']:.3f}, AUC {tm['ROC-AUC']:.3f}), but it also raises a false alarm on "
        f"<b>{tm['FPR']:.1%}</b> of normal flows. The most influential features are "
        f"<b>{top_feats[0]}</b>, <b>{top_feats[1]}</b> and <b>{top_feats[2]}</b>. "
        "See the <i>Performance</i> and <i>Notes &amp; Limitations</i> tabs for details.",
    )

# ------------------------------------------------------------------ background
with tabs[1]:
    st.header("Background of the study")
    st.write(
        "As organizations rely more on networked services, the volume and variety of cyberattacks keeps "
        "growing. A network intrusion detection system monitors traffic and raises an alert when it "
        "observes behavior that suggests an attack. Traditional **signature-based** systems compare traffic "
        "against known attack patterns; they are accurate for known threats but cannot recognize new or "
        "modified attacks. **Anomaly-based** systems instead learn what normal traffic looks like and flag "
        "deviations, which helps against unseen attacks but tends to produce more false alarms."
    )
    st.write(
        "Machine learning offers a practical middle path: a model can learn the statistical difference "
        "between normal and malicious flows directly from labelled examples. Many studies have applied "
        "classifiers such as decision trees, support vector machines, neural networks and ensemble methods "
        "to intrusion detection. Their results depend heavily on the dataset used. Older benchmarks such as "
        "KDD Cup 99 and NSL-KDD are widely criticized for outdated traffic and attack types, which motivated "
        "newer datasets such as **UNSW-NB15**, built from more modern traffic and attack behaviors."
    )
    st.write(
        "The **Random Forest** algorithm is a strong candidate for this task. It builds many decision trees "
        "on random samples of the data and random subsets of features, then combines their votes. This "
        "reduces the overfitting of a single tree, copes well with mixed numeric and categorical features "
        "and noisy data, needs little feature scaling, and reports which features matter most, which is "
        "useful for understanding what separates attacks from normal traffic."
    )

    st.subheader("Reference")
    st.markdown(
        '<span class="small">Moustafa, N., & Slay, J. (2015). UNSW-NB15: a comprehensive data set for network '
        "intrusion detection systems (UNSW-NB15 network data set). <i>2015 Military Communications and "
        "Information Systems Conference (MilCIS)</i>, IEEE.</span>",
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------- dataset
with tabs[2]:
    st.header("Dataset: UNSW-NB15")
    st.write(
        "UNSW-NB15 was created by the Australian Centre for Cyber Security (ACCS) at UNSW Canberra. "
        "Its authors used the IXIA PerfectStorm tool to generate a mix of realistic normal activity and "
        "synthetic, contemporary attack behavior, captured the traffic, and extracted flow-level features "
        "with the Argus and Bro-IDS tools. Each record describes one network flow and carries a label."
    )

    c = st.columns(3)
    c[0].metric("Training partition", f"{ov['rows']:,}")
    c[0].caption(f"{train_attack_pct:.0f}% attack")
    c[1].metric("Testing partition", f"{ov['test_rows']:,}")
    c[1].caption(f"{test_attack_pct:.0f}% attack")
    c[2].metric("Input features", n_raw)
    c[2].caption("after dropping id and attack_cat")

    st.subheader("Attack families")
    st.write(f"The {ov['attack_types']} attack categories in the training partition are shown below. "
             "Normal traffic makes up the rest of the records.")
    counts = pd.Series(ov["attack_counts"]).sort_values(ascending=False)
    st.dataframe(counts.rename("Training records").to_frame())

    st.subheader("Feature groups")
    st.markdown(
        """
| Group | What it describes |
|---|---|
| **Flow** | Protocol and addressing information of the flow |
| **Basic** | Duration, packet and byte counts, TTL values, loss and load |
| **Content** | TCP sequence numbers, window sizes, mean packet size, HTTP-related counts |
| **Time** | Inter-packet arrival times, jitter, round-trip times |
| **Additional generated** | Connection counts over recent flows (the `ct_*` features), grouped by state, service, address and port |

Three features are categorical (`proto`, `service`, `state`); the rest are numeric. The
dataset also contains `attack_cat` (the attack family) and `label` (0 = normal, 1 = attack).
"""
    )

    st.subheader("A note on the two partitions")
    callout(
        "Why validation and test results differ",
        f"The official training partition is {train_attack_pct:.0f}% attack while the testing partition is "
        f"{test_attack_pct:.0f}% attack, and the two were not drawn identically. A model that scores very well "
        "on a split of the training data can therefore score noticeably lower on the testing partition. "
        "This study reports both so the difference is visible.",
    )

    st.subheader("Sample of the training data")
    st.dataframe(load_csv("sample_train.csv"))

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
        st.session_state.setdefault(f"in_{name}", meta["defaults"][name])
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
    st.write("Enter the details of a network flow, or load a real record from the test partition.")
    st.button("Load a random example", on_click=randomize)

    cols = st.columns(3)
    for i, (name, help_txt) in enumerate(FIELDS.items()):
        with cols[i % 3]:
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
            st.error(f"**Attack**: predicted attack probability {prob[0]:.1%}")
        else:
            st.success(f"**Normal**: predicted attack probability {prob[0]:.1%}")
        st.progress(float(prob[0]))
        true = st.session_state.get("example_label")
        if true is not None:
            st.caption(f"True label of the loaded example: {'Attack' if true else 'Normal'} "
                       "(editing the fields can change the outcome).")
        st.caption("Features not shown use the loaded example's values, or typical (median) values.")


def csv_upload():
    st.write(
        "Upload a CSV containing the raw UNSW-NB15 feature columns. "
        "The columns id, attack_cat, and label are optional."
    )

    sample = load_csv("sample_input.csv")

    st.download_button(
        "Download Sample CSV",
        data=sample.to_csv(index=False).encode("utf-8"),
        file_name="sample_input.csv",
        mime="application/octet-stream"
    )

    uploaded_file = st.file_uploader(
        "Upload Network Traffic CSV",
        type=None,
        accept_multiple_files=False,
        key="network_csv"
    )

    if uploaded_file is None:
        st.info("Select a CSV file to begin.")
        return

    if not uploaded_file.name.lower().endswith(".csv"):
        st.error("Please select a valid .csv file.")
        return

    try:
        data = pd.read_csv(uploaded_file)

        if data.empty:
            st.error("The uploaded CSV contains no records.")
            return

        missing = [
            col for col in meta["columns"]
            if col not in data.columns
        ]

        if missing:
            st.error(f"Missing required columns: {missing}")
            return

        st.success("CSV uploaded successfully!")

        pred, prob = predict(data)

        out = data.copy()
        out["predicted_label"] = pred
        out["attack_probability"] = prob.round(4)

        c1, c2, c3 = st.columns(3)

        c1.metric("Records", f"{len(out):,}")
        c2.metric("Predicted Attacks", f"{int(pred.sum()):,}")
        c3.metric("Predicted Normal", f"{int((pred == 0).sum()):,}")

        if "label" in data.columns:
            y = data["label"].astype(int)

            tn, fp, fn, tp = confusion_matrix(
                y, pred, labels=[0, 1]
            ).ravel()

            m1, m2, m3 = st.columns(3)

            m1.metric(
                "Accuracy",
                f"{accuracy_score(y, pred):.2%}"
            )

            m2.metric(
                "F1-Score",
                f"{f1_score(y, pred):.3f}"
            )

            fpr = fp / (fp + tn) if (fp + tn) else 0

            m3.metric(
                "False Positive Rate",
                f"{fpr:.2%}"
            )

        st.dataframe(out.head(200), use_container_width=True)

        st.download_button(
            "Download Predictions",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error processing CSV: {e}")
      
# ----------------------------------------------------------------- performance
with tabs[4]:
    st.header("Model performance")
    st.write(f"Results on the official UNSW-NB15 test partition ({ov['test_rows']:,} flows, never used for "
             "training or tuning).")
    cols = st.columns(6)
    for col, (label, key) in zip(cols, [("Accuracy", "Accuracy"), ("Precision", "Precision"), ("Recall", "Recall"),
                                         ("F1-score", "F1-Score"), ("ROC-AUC", "ROC-AUC"), ("FPR", "FPR")]):
        col.metric(label, f"{tm[key]:.3f}")

    callout(
        "How to read these numbers",
        f"<b>Recall</b> ({tm['Recall']:.1%}) is the share of real attacks the model catches. "
        f"<b>False positive rate</b> ({tm['FPR']:.1%}) is the share of normal flows wrongly flagged as attacks. "
        f"<b>Precision</b> ({tm['Precision']:.1%}) is the share of alerts that are real attacks. "
        f"On a validation split of the training data the same model reached {val['Accuracy']:.1%} accuracy with a "
        f"{val['FPR']:.1%} false positive rate, so the lower test score reflects a difference between the two "
        "partitions rather than a defect in the model.",
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Confusion matrix")
        st.pyplot(cm_fig())
    with right:
        st.subheader("ROC curve")
        st.pyplot(roc_fig())

    st.subheader("Feature importance")
    st.pyplot(importance_fig())

    st.subheader("Model comparison")
    st.caption("All three models were evaluated on the same test partition.")
    st.dataframe(pd.DataFrame(meta["comparison"]).set_index("Model"))

    st.subheader("Sample test-set predictions")
    st.dataframe(load_csv("sample_predictions.csv"))

# ----------------------------------------------------------------------- notes
with tabs[5]:
    st.header("Important notes and limitations")
    callout(
        "Please read before relying on any result",
        "This application is an academic demonstration. It should not be used to make real security decisions.",
    )
    st.subheader("Limitations of the model")
    st.markdown(
        f"""
- **High false positive rate.** About {tm['FPR']:.0%} of normal flows in the test partition are flagged as attacks. In a real network, this would overwhelm analysts with false alarms.
- **Validation vs. test gap.** Performance on the validation split ({val['Accuracy']:.1%} accuracy) is much higher than on the test partition ({tm['Accuracy']:.1%}), because the two partitions differ in composition. Neither number alone describes real-world performance.
- **Binary decision only.** The model says Normal or Attack. It does not identify the attack family.
- **Fixed decision threshold.** Predictions use a probability threshold of 0.5. A different threshold would trade recall against false positives.
"""
    )
    st.subheader("Limitations of the data")
    st.markdown(
        """
- **Synthetic and dated.** UNSW-NB15 was generated in a laboratory in 2015. Attack techniques and normal traffic patterns have since changed.
- **Single dataset.** The model was not tested on other datasets or on live traffic, so its ability to generalize to other networks is unknown.
- **Offline, flow-level features.** The features are computed from complete flows after capture; the model is not a real-time sensor.
"""
    )
    st.subheader("Using this application")
    st.markdown(
        """
- The manual-entry form exposes nine of the model's input features. The remaining features keep the values of the loaded example or typical (median) values, so hand-entered flows may not resemble real traffic.
- The example records come from the official test partition, so they were never used to train the model.
- Uploaded CSV files are processed in your session only. The app does not store them.
"""
    )
    st.subheader("Possible future work")
    st.markdown(
        """
- Compare against other algorithms (for example Decision Tree, Gradient Boosting) under the same protocol.
- Evaluate threshold tuning and resampling to lower the false positive rate.
- Test on newer datasets to measure generalization.
- Extend to multi-class detection of the individual attack families.
"""
    )
