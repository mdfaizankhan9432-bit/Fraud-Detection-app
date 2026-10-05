"""
Fraud Detection - Streamlit app (deployment file)

Run locally :  streamlit run app.py
Needs these files in the SAME folder as this app.py (all created by 03_model_building.ipynb):
    best_fraud_model.joblib, model_metadata.json, requirements.txt,
    sample_transactions.csv, Cleaned_Dataset.csv
The full GitHub + Streamlit Cloud steps are inside the app, in the "Deployment guide" tab.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Fraud Detection App", page_icon="🛡️", layout="wide")

BASE = Path(__file__).resolve().parent
MODEL_FILE = BASE / "best_fraud_model.joblib"
META_FILE = BASE / "model_metadata.json"
SAMPLE_FILE = BASE / "sample_transactions.csv"
DATA_FILE = BASE / "Cleaned_Dataset.csv"
REQ_FILE = BASE / "requirements.txt"

# Used only if model_metadata.json is missing. Normally these come from the metadata file.
FALLBACK_INPUT_COLUMNS = ["step", "type", "amount", "oldbalanceOrg", "newbalanceOrig",
                          "oldbalanceDest", "newbalanceDest"]
FALLBACK_FEATURE_COLUMNS = FALLBACK_INPUT_COLUMNS + ["balance_change_orig", "balance_change_dest", "hour_of_day"]
FALLBACK_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]


# --------------------------------------------------------------------------------------
# Feature engineering - this MUST stay identical to add_features() in 03_model_building.ipynb
# --------------------------------------------------------------------------------------
def add_features(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["balance_change_orig"] = d["oldbalanceOrg"] - d["newbalanceOrig"]
    d["balance_change_dest"] = d["newbalanceDest"] - d["oldbalanceDest"]
    d["hour_of_day"] = d["step"] % 24
    return d


# --------------------------------------------------------------------------------------
# Loading (cached, so files are read only once)
# --------------------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


@st.cache_data
def load_metadata():
    if META_FILE.exists():
        with open(META_FILE) as f:
            return json.load(f)
    return {}


@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_FILE) if DATA_FILE.exists() else None


@st.cache_data
def load_sample():
    return pd.read_csv(SAMPLE_FILE) if SAMPLE_FILE.exists() else None


meta = load_metadata()
INPUT_COLUMNS = meta.get("input_columns", FALLBACK_INPUT_COLUMNS)
FEATURE_COLUMNS = meta.get("feature_columns", FALLBACK_FEATURE_COLUMNS)
TYPES = meta.get("transaction_types", FALLBACK_TYPES)

if not MODEL_FILE.exists():
    st.title("🛡️ Fraud Detection App")
    st.error("`best_fraud_model.joblib` was not found next to `app.py`.")
    st.markdown(
        "Run all cells of **03_model_building.ipynb** first. It creates a `deployment_files/` folder with the model. "
        "Put this `app.py` inside that folder and start the app again."
    )
    st.stop()

try:
    model = load_model()
except Exception as e:  # most common cause: scikit-learn / lightgbm version mismatch
    st.title("🛡️ Fraud Detection App")
    st.error("The model file could not be loaded.")
    st.markdown(
        "This almost always means the library versions here are different from the ones used for training. "
        "Install the versions from `requirements.txt` (they were written by the notebook)."
    )
    st.code(str(e))
    st.stop()


def score(raw: pd.DataFrame) -> np.ndarray:
    """raw has the 7 input columns -> returns fraud probability for every row."""
    feats = add_features(raw[INPUT_COLUMNS])[FEATURE_COLUMNS]
    return model.predict_proba(feats)[:, 1]


def money(x) -> str:
    return f"{x:,.0f}"


# --------------------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------------------
st.sidebar.title("🛡️ Fraud Detection")
st.sidebar.caption("Mobile-money transaction fraud checker (class project).")
st.sidebar.markdown(f"**Model:** {meta.get('final_model', 'saved pipeline')}")
if meta.get("final_test_scores"):
    s = meta["final_test_scores"]
    st.sidebar.markdown(
        f"**Test scores**  \nPrecision {s['Precision']:.3f} · Recall {s['Recall']:.3f} · F1 {s['F1 Score']:.3f}"
    )
default_threshold = float(meta.get("decision_threshold", 0.5))
threshold = st.sidebar.slider(
    "Alert threshold", 0.05, 0.95, default_threshold, 0.05,
    help="A transaction is flagged as fraud when its fraud probability is at or above this value. "
         "The model's reported scores were measured at 0.50.",
)
if abs(threshold - default_threshold) > 1e-9:
    st.sidebar.warning("Reported scores were measured at 0.50. Other thresholds trade recall against false alarms.")

st.title("🛡️ Fraud Detection App")
tab_single, tab_batch, tab_info, tab_deploy = st.tabs(
    ["🔍 Check a transaction", "📂 Check many (CSV)", "📊 Data & model info", "🚀 Deployment guide"]
)

# --------------------------------------------------------------------------------------
# Tab 1 - single transaction
# --------------------------------------------------------------------------------------
EXAMPLES = {
    "normal": {"f_type": "PAYMENT", "f_step": 10, "f_amount": 1200.0, "f_oldO": 25000.0,
               "f_newO": 23800.0, "f_oldD": 0.0, "f_newD": 0.0},
    "fraud": {"f_type": "TRANSFER", "f_step": 1, "f_amount": 181.0, "f_oldO": 181.0,
              "f_newO": 0.0, "f_oldD": 0.0, "f_newD": 0.0},
}
for k, v in EXAMPLES["normal"].items():
    st.session_state.setdefault(k, v)


def load_example(name: str):
    for k, v in EXAMPLES[name].items():
        if k == "f_type" and v not in TYPES:
            continue
        st.session_state[k] = v


def consistency_issues(row: dict) -> list:
    """Checks whether the typed numbers can really belong to ONE real transaction.
    If any of these fail, the inputs are contradictory and the model's answer should not be trusted."""
    issues = []
    tol = 0.01
    old_o, new_o = row["oldbalanceOrg"], row["newbalanceOrig"]
    old_d, new_d = row["oldbalanceDest"], row["newbalanceDest"]
    amt, t = row["amount"], row["type"]

    # sender side: CASH_IN adds money to the sender's account, every other type takes money out
    if t == "CASH_IN":
        if abs((new_o - old_o) - amt) > tol:
            issues.append(f"For CASH_IN the sender's balance should go UP by the amount ({amt:,.2f}), "
                          f"but it changed by {new_o - old_o:+,.2f}.")
    else:
        if abs((old_o - new_o) - amt) > tol:
            issues.append(f"The sender's balance should go DOWN by the amount ({amt:,.2f}), "
                          f"but it changed by {new_o - old_o:+,.2f}.")

    # receiver side: the receiver can never gain more than the amount, and cannot lose money on a transfer / cash-out
    gain = new_d - old_d
    if gain > amt + tol:
        issues.append(f"The receiver's balance went up by {gain:,.2f}, which is more than the amount ({amt:,.2f}).")
    if t in ("TRANSFER", "CASH_OUT") and gain < -tol:
        issues.append(f"The receiver's balance went DOWN by {-gain:,.2f} on a {t}, which should not happen.")
    return issues


def sanity_notes(row: dict) -> list:
    """Softer observations. The numbers are consistent, but these patterns are worth a second look.
    These are NOT how the model decides - just plain-language hints."""
    notes = []
    if row["oldbalanceOrg"] > 0 and row["newbalanceOrig"] == 0:
        notes.append("The sender's account was emptied completely (new balance is 0).")
    if row["type"] in ("TRANSFER", "CASH_OUT") and row["newbalanceDest"] - row["oldbalanceDest"] <= 0:
        notes.append("The receiver's balance did not go up, although money was supposedly sent to them.")
    if row["type"] != "CASH_IN" and row["amount"] > row["oldbalanceOrg"]:
        notes.append("The amount is larger than the sender's balance before the transaction.")
    return notes


with tab_single:
    st.subheader("Enter one transaction")
    b1, b2, _ = st.columns([1, 1, 3])
    b1.button("Load a normal example", on_click=load_example, args=("normal",))
    b2.button("Load a fraud-like example", on_click=load_example, args=("fraud",))

    c1, c2, c3 = st.columns(3)
    with c1:
        tx_type = st.selectbox("Transaction type", TYPES, key="f_type")
        step = st.number_input("Step (hours since start of data; 1 step = 1 hour)", min_value=0, step=1, key="f_step",
                               help="The app turns this into the hour of the day (step modulo 24), "
                                    "exactly like the notebook.")
        amount = st.number_input("Amount", min_value=0.0, step=100.0, format="%.2f", key="f_amount")
    with c2:
        old_o = st.number_input("Sender balance BEFORE", min_value=0.0, step=100.0, format="%.2f", key="f_oldO")
        new_o = st.number_input("Sender balance AFTER", min_value=0.0, step=100.0, format="%.2f", key="f_newO")
    with c3:
        old_d = st.number_input("Receiver balance BEFORE", min_value=0.0, step=100.0, format="%.2f", key="f_oldD")
        new_d = st.number_input("Receiver balance AFTER", min_value=0.0, step=100.0, format="%.2f", key="f_newD")

    if st.button("Check this transaction", type="primary"):
        row = {"step": int(step), "type": tx_type, "amount": float(amount),
               "oldbalanceOrg": float(old_o), "newbalanceOrig": float(new_o),
               "oldbalanceDest": float(old_d), "newbalanceDest": float(new_d)}
        prob = float(score(pd.DataFrame([row]))[0])
        is_fraud = prob >= threshold

        issues = consistency_issues(row)
        if issues:
            st.warning("**Input check failed - these numbers do not add up as one real transaction, "
                       "so the result below is NOT reliable.**\n\n" + "\n".join(f"- {i}" for i in issues) +
                       "\n\nFix the numbers above and check again.")
        else:
            st.info("Input check passed: the numbers are consistent with one real transaction.")

        r1, r2 = st.columns(2)
        r1.metric("Fraud probability", f"{prob:.1%}")
        r2.metric("Decision", "FRAUD ALERT" if is_fraud else "Looks legitimate")
        st.progress(min(max(prob, 0.0), 1.0))
        if issues:
            st.caption("Result shown only for reference, because the input check failed.")
        elif is_fraud:
            st.error("This transaction should be sent for manual review.")
        else:
            st.success("No fraud alert for this transaction.")

        with st.expander("What the model actually saw (after feature engineering)"):
            st.dataframe(add_features(pd.DataFrame([row]))[FEATURE_COLUMNS])

        notes = sanity_notes(row)
        if notes:
            st.markdown("**Worth a second look (simple checks, not the model's reasoning):**")
            for n in notes:
                st.markdown(f"- {n}")

    st.caption("Demo project. A real bank would combine a model like this with human review and more data.")

# --------------------------------------------------------------------------------------
# Tab 2 - batch (CSV)
# --------------------------------------------------------------------------------------
with tab_batch:
    st.subheader("Check many transactions at once")
    st.markdown("Upload a CSV with these columns: `" + "`, `".join(INPUT_COLUMNS) + "`. "
                "Extra columns are ignored. If the file also has an `isFraud` column, the app compares its answers with it.")

    sample_df = load_sample()
    if sample_df is not None:
        st.download_button("Download the sample file (20 real test transactions)",
                           sample_df.to_csv(index=False).encode("utf-8"),
                           file_name="sample_transactions.csv", mime="text/csv")

    uploaded = st.file_uploader("Upload your CSV", type="csv")
    use_sample = False
    if sample_df is not None:
        use_sample = st.checkbox("No file handy? Use the built-in sample file")

    data = None
    if uploaded is not None:
        data = pd.read_csv(uploaded)
    elif use_sample:
        data = sample_df.copy()

    if data is not None:
        missing = [c for c in INPUT_COLUMNS if c not in data.columns]
        if missing:
            st.error("These required columns are missing: " + ", ".join(missing))
        else:
            work = data.copy()
            for c in INPUT_COLUMNS:
                if c != "type":
                    work[c] = pd.to_numeric(work[c], errors="coerce")
            bad = work[INPUT_COLUMNS].isna().any(axis=1)
            if bad.any():
                st.warning(f"{int(bad.sum())} row(s) have missing or non-numeric values and were skipped.")
            work = work[~bad].copy()

            if work.empty:
                st.error("No valid rows to score.")
            else:
                work["fraud_probability"] = score(work)
                work["prediction"] = np.where(work["fraud_probability"] >= threshold, "FRAUD", "Legit")
                work["input_check"] = [
                    "OK" if not consistency_issues(r) else "INCONSISTENT"
                    for r in work[INPUT_COLUMNS].to_dict("records")
                ]

                k1, k2, k3 = st.columns(3)
                k1.metric("Transactions checked", len(work))
                k2.metric("Flagged as fraud", int((work["prediction"] == "FRAUD").sum()))
                k3.metric("Flag rate", f"{(work['prediction'] == 'FRAUD').mean():.1%}")

                n_bad_inputs = int((work["input_check"] != "OK").sum())
                if n_bad_inputs:
                    st.warning(f"{n_bad_inputs} row(s) have numbers that do not add up (column `input_check`). "
                               "Their predictions are not reliable.")

                if "isFraud" in work.columns:
                    actual = work["isFraud"].astype(int)
                    predicted = (work["prediction"] == "FRAUD").astype(int)
                    st.markdown(f"**Compared with the `isFraud` column:** {int((actual == predicted).sum())} of "
                                f"{len(work)} correct ({(actual == predicted).mean():.1%}).")
                    st.dataframe(pd.crosstab(actual.map({0: "Actual legit", 1: "Actual fraud"}),
                                             predicted.map({0: "Predicted legit", 1: "Predicted fraud"})))

                show = work.sort_values("fraud_probability", ascending=False)
                st.dataframe(show)
                st.download_button("Download results as CSV", show.to_csv(index=False).encode("utf-8"),
                                   file_name="fraud_predictions.csv", mime="text/csv")

# --------------------------------------------------------------------------------------
# Tab 3 - data & model info
# --------------------------------------------------------------------------------------
COLUMN_INFO = {
    "step": "Time in hours since the start of the data (1 step = 1 hour)",
    "type": "Kind of transaction: CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER",
    "amount": "Amount of the transaction",
    "nameOrig": "ID of the sender (not used by the model - almost unique per row)",
    "oldbalanceOrg": "Sender's balance before the transaction",
    "newbalanceOrig": "Sender's balance after the transaction",
    "nameDest": "ID of the receiver (not used by the model)",
    "oldbalanceDest": "Receiver's balance before the transaction",
    "newbalanceDest": "Receiver's balance after the transaction",
    "isFraud": "Target: 1 = fraud, 0 = legitimate",
}

with tab_info:
    ds_info = meta.get("dataset", {})
    st.subheader("Dataset")
    if ds_info:
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Rows", f"{ds_info['rows']:,}")
        d2.metric("Columns", ds_info["columns"])
        d3.metric("Fraud share", f"{ds_info['fraud_percent']}%")
        d4.metric("Train / test rows", f"{ds_info['train_rows']:,} / {ds_info['test_rows']:,}")

    st.markdown("**What each column means**")
    st.dataframe(pd.DataFrame({"column": list(COLUMN_INFO), "meaning": list(COLUMN_INFO.values())}))

    ds = load_dataset()
    if ds is not None:
        g1, g2 = st.columns(2)
        with g1:
            st.markdown("**Transactions per type**")
            st.bar_chart(ds["type"].value_counts())
        with g2:
            st.markdown("**Fraud share per type (%)**")
            st.bar_chart((ds.groupby("type")["isFraud"].mean() * 100).round(2))
        with st.expander("Preview of the cleaned dataset (first 50 rows)"):
            st.dataframe(ds.head(50))
    elif ds_info.get("type_counts"):
        st.markdown("**Transactions per type**")
        st.bar_chart(pd.Series(ds_info["type_counts"]))

    st.divider()
    st.subheader("Model")
    st.markdown(
        "**How a prediction is made:** the 7 inputs go through `add_features()` (adds the sender's balance change, "
        "the receiver's balance change and the hour of the day) → scaling and one-hot encoding of `type` → the model "
        "returns a fraud probability → the alert threshold turns it into FRAUD / Legit."
    )
    if meta:
        m1, m2, m3 = st.columns(3)
        m1.metric("Final model", meta.get("final_model", "-"))
        m2.metric("Tuning used", "Yes" if meta.get("tuning_used") else "No (default settings were as good)")
        m3.metric("Picked by", "Manual choice" if meta.get("manual_model_choice") else "5-fold CV F1")
        if meta.get("manual_model_choice") and meta.get("cv_top_model"):
            st.caption(f"Cross-validation top model was {meta['cv_top_model']}; "
                       f"{meta['manual_model_choice']} was chosen manually because the scores are very close.")
        if meta.get("tuned_params"):
            st.markdown("Tuned settings: `" + json.dumps(meta["tuned_params"]) + "`")

        if meta.get("cv_results"):
            st.markdown("**Cross-validation on the training data (used to choose the model)**")
            st.dataframe(pd.DataFrame(meta["cv_results"]).rename(columns={
                "model": "Model", "cv_f1_mean": "CV F1 (mean)", "cv_f1_std": "CV F1 (std)"}))
        if meta.get("test_results_all_models"):
            st.markdown("**Test-set scores of all models (reference only, not used for choosing)**")
            st.dataframe(pd.DataFrame(meta["test_results_all_models"]).rename(columns={
                "model": "Model", "precision": "Precision", "recall": "Recall", "f1": "F1", "roc_auc": "ROC-AUC"}))

        fs, cm = meta.get("final_test_scores"), meta.get("confusion_matrix")
        if fs and cm:
            st.markdown("**Final model on the untouched test set**")
            f1c, f2c, f3c, f4c = st.columns(4)
            f1c.metric("Precision", f"{fs['Precision']:.3f}")
            f2c.metric("Recall", f"{fs['Recall']:.3f}")
            f3c.metric("F1", f"{fs['F1 Score']:.3f}")
            f4c.metric("ROC-AUC", f"{fs['ROC-AUC']:.4f}")
            st.dataframe(pd.DataFrame(
                [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]],
                index=["Actual legit", "Actual fraud"], columns=["Predicted legit", "Predicted fraud"]))

        fin = meta.get("financial")
        if fin:
            st.markdown("**Financial impact on the test set** "
                        f"(assumed review cost: {fin['review_cost_per_flag']} per flagged transaction)")
            n1, n2, n3, n4 = st.columns(4)
            n1.metric("Loss without any model", money(fin["loss_without_model"]))
            n2.metric("Loss with this model", money(fin["loss_with_model"]))
            n3.metric("Net saving", money(fin["net_saving"]))
            n4.metric("Fraud money stopped", f"{fin['fraud_stopped'] / fin['total_fraud_amount']:.1%}"
                      if fin["total_fraud_amount"] else "-")
            st.caption("Loss with the model = fraud that slipped through + review cost of every flagged transaction.")

        st.caption(f"Model created on {meta.get('created_on', '-')} · Python {meta.get('python_version', '-')} · "
                   f"libraries: {json.dumps(meta.get('library_versions', {}))}")
        st.caption("Scores on this dataset are very high because its fraud follows a clean pattern. "
                   "Real-world fraud is messier, so live performance would be lower.")
    else:
        st.info("`model_metadata.json` was not found, so model details cannot be shown.")

# --------------------------------------------------------------------------------------
# Tab 4 - deployment guide
# --------------------------------------------------------------------------------------
with tab_deploy:
    st.subheader("How to deploy this app (GitHub + Streamlit Community Cloud)")

    st.markdown("### Step 0 - What you need")
    st.markdown(
        "- A free GitHub account and a free Streamlit Community Cloud account (you can sign in to Streamlit with GitHub)\n"
        "- The `deployment_files/` folder created by **03_model_building.ipynb**, with this `app.py` placed inside it"
    )

    st.markdown("### Step 1 - Check the files")
    st.markdown("Your project folder should look like this (all files in the same place):")
    st.code(
        "fraud-detection-app/\n"
        "├── app.py                    <- this file\n"
        "├── requirements.txt          <- libraries + exact versions (written by the notebook)\n"
        "├── best_fraud_model.joblib   <- the trained model\n"
        "├── model_metadata.json       <- scores, dataset info, settings\n"
        "├── sample_transactions.csv   <- 20 real test transactions for the CSV tab\n"
        "└── Cleaned_Dataset.csv       <- data shown in the 'Data & model info' tab",
        language="text",
    )
    if REQ_FILE.exists():
        st.markdown("Your current `requirements.txt`:")
        st.code(REQ_FILE.read_text(), language="text")

    st.markdown("### Step 2 - Test it on your own computer first")
    st.code("pip install -r requirements.txt\nstreamlit run app.py", language="bash")
    st.markdown("A browser tab opens at `http://localhost:8501`. Try both examples and the sample CSV. "
                "If everything works here, it will work online.")

    st.markdown("### Step 3 - Upload to GitHub")
    st.markdown("**Option A - no commands (easiest):** on github.com click **New repository**, give it a name "
                "(e.g. `fraud-detection-app`), keep it **Public**, create it, then click **Add file → Upload files**, "
                "drag in all 6 files and press **Commit changes**.")
    st.markdown("**Option B - with Git:** open a terminal inside the project folder and run:")
    st.code(
        "git init\n"
        "git add .\n"
        'git commit -m "Fraud detection Streamlit app"\n'
        "git branch -M main\n"
        "git remote add origin https://github.com/<your-username>/fraud-detection-app.git\n"
        "git push -u origin main",
        language="bash",
    )
    st.markdown("Make sure the files sit in the **root** of the repository (not inside an extra folder), "
                "or remember the folder name for Step 4.")

    st.markdown("### Step 4 - Deploy on Streamlit Community Cloud")
    st.markdown(
        "1. Go to **share.streamlit.io** and sign in with GitHub.\n"
        "2. Click **Create app** and choose to deploy from an existing GitHub repository.\n"
        "3. Pick your repository and the branch `main`.\n"
        "4. **Main file path:** `app.py` (or `your-folder/app.py` if you kept it inside a folder).\n"
        "5. Open **Advanced settings** and choose the same Python version you trained with "
        f"(this model was trained with Python {meta.get('python_version', 'the one in your notebook')}).\n"
        "6. Click **Deploy**. The first build takes a few minutes while the libraries install. "
        "You then get a public link like `https://your-app-name.streamlit.app` that you can share."
    )
    st.caption("Button names on the Streamlit site can change slightly over time, but the flow stays the same.")

    st.markdown("### Step 5 - Updating the app later")
    st.markdown("Retrained the model? Run the notebook again, replace the files in the repository "
                "(`best_fraud_model.joblib`, `model_metadata.json`, `requirements.txt`, ...) and commit. "
                "Streamlit redeploys automatically after every commit.")

    st.markdown("### Common problems")
    st.markdown(
        "- **Model could not be loaded / version error:** the libraries online differ from the ones used for training. "
        "Use the `requirements.txt` produced by the notebook and the same Python version.\n"
        "- **FileNotFoundError:** a file is missing from the repo or is in a different folder than `app.py`.\n"
        "- **App build fails on `lightgbm`:** check the spelling and the version in `requirements.txt`.\n"
        "- **Predictions look different from the notebook:** `add_features()` in this file must match the notebook "
        "exactly; do not change one without the other.\n"
        "- **App went to sleep:** free apps sleep after a period of no visitors. Opening the link wakes them up."
    )
