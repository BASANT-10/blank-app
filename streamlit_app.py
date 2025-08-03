# classifier_word_metrics.py
import streamlit as st
import pandas as pd
import re, ast

st.set_page_config(page_title="Classifier Word Metrics", layout="wide")
st.title("📈 Classifier Word Metrics")

# ───────────────────────── STEP 1 – UPLOAD DATA ─────────────────────────
file = st.file_uploader("📁 **Step 1 – Upload CSV** (must contain a text column and 0/1 ground‑truth column)",
                        type="csv")

if not file:
    st.info("Upload a CSV to continue.")
    st.stop()

df_raw = pd.read_csv(file)
st.success("Dataset loaded.")
with st.expander("Preview data"):
    st.dataframe(df_raw.head())

# ───────────────────────── STEP 2 – BASIC SETTINGS ──────────────────────
default_tactics = {
    "urgency_marketing":  ["now", "today", "limited", "hurry", "exclusive"],
    "social_proof":       ["bestseller", "popular", "trending", "recommended"],
    "discount_marketing": ["sale", "discount", "deal", "free", "offer"],
}

tactic = st.selectbox("🎯 **Step 2 – Choose a tactic to evaluate**", list(default_tactics.keys()))
text_col  = st.selectbox("📝 Text column",  df_raw.columns)
label_col = st.selectbox("🏷️  Ground‑truth label column (1 = positive, 0 = negative)", df_raw.columns)

# Pre‑clean text once
def clean(txt: str) -> str:
    return re.sub(r"[^A-Za-z0-9\s]", "", str(txt).lower())

df = df_raw.copy()
df["_clean"] = df[text_col].astype(str).apply(clean)

# ───────────────────────── STEP 3 – DICTIONARY SETUP ────────────────────
st.header("🔧 Step 3 – Dictionary setup")

dict_mode = st.radio("Choose one method:", ["🧠 Generate from data", "📥 Provide custom dictionary"])

# Clear session flags if user switches modes or uploads new file
if "prev_mode" in st.session_state and st.session_state.prev_mode != dict_mode:
    for key in ("gen_ready", "dict_ready", "dictionary", "auto_dict", "top_words"):
        st.session_state.pop(key, None)
st.session_state.prev_mode = dict_mode  # remember selection

# ---------- Option A: Generate from data ----------
if dict_mode == "🧠 Generate from data":

    if st.button("Generate dictionary from dataset"):
        freq = pd.Series(" ".join(df["_clean"]).split()).value_counts()
        top_words = freq[freq > 1].head(20)
        st.session_state.top_words  = top_words
        st.session_state.auto_dict  = {tactic: set(top_words.index)}
        st.session_state.gen_ready  = True      # flag that generation finished

    # Show editor & save button after generation
    if st.session_state.get("gen_ready"):
        st.write("Top keywords:")
        st.dataframe(st.session_state.top_words)

        st.write("Auto‑generated dictionary:", st.session_state.auto_dict)

        dict_text = st.text_area(
            "✏️ Edit dictionary (Python dict syntax)",
            value=str(st.session_state.auto_dict),
            height=150,
            key="dict_edit_box",
        )

        if st.button("Save dictionary"):        # this button now persists
            try:
                st.session_state.dictionary = ast.literal_eval(dict_text)
                st.session_state.dict_ready = True
                st.success("Dictionary saved.")
            except Exception:
                st.error("Bad format – please correct and click Save again.")

# ---------- Option B: Provide custom dictionary ----------
elif dict_mode == "📥 Provide custom dictionary":
    template = f"{{'{tactic}': {{'keyword1', 'keyword2'}}}}"
    dict_text = st.text_area("Paste your dictionary here", value=template, height=150)
    if st.button("Save custom dictionary"):
        try:
            st.session_state.dictionary = ast.literal_eval(dict_text)
            st.session_state.dict_ready = True
            st.success("Custom dictionary saved.")
        except Exception:
            st.error("Bad format – please correct and click Save again.")

# Block evaluation until dictionary is saved
if not st.session_state.get("dict_ready"):
    st.info("➡️  Finish Step 3 (save a dictionary) to unlock evaluation.")
    st.stop()

dictionary = st.session_state.dictionary
keywords   = list(dictionary[tactic])

# ───────────────────────── STEP 4 – RUN EVALUATION ──────────────────────
st.header("🚀 Step 4 – Run evaluation")

if st.button("Start evaluation"):
    # Predictions
    df["_pred"] = df["_clean"].apply(lambda t: 1 if any(k in t.split() for k in keywords) else 0)
    df["_matches"] = df["_clean"].apply(lambda t: [k for k in keywords if k in t.split()])

    # --- Classification results (True Positives) ---
    st.subheader("📝 True‑positive rows")
    tp_df = df[(df[label_col] == 1) & (df["_pred"] == 1)]
    more = st.checkbox("Show all true‑positives", False)
    st.dataframe(tp_df if more else tp_df.head(20))

    accuracy = (df[label_col] == df["_pred"]).mean()
    st.metric("Overall accuracy", f"{accuracy*100:.1f}%")

    # --- Keyword impact analysis ---
    st.header("📊 Keyword impact analysis")

    rows = []
    for kw in keywords:
        mask = df["_clean"].str.contains(fr"\b{kw}\b")
        tp = ((mask)  & (df[label_col]==1)).sum()
        fp = ((mask)  & (df[label_col]==0)).sum()
        fn = ((~mask) & (df[label_col]==1)).sum()
        prec = tp/(tp+fp) if tp+fp else 0
        rec  = tp/(tp+fn) if tp+fn else 0
        f1   = 2*prec*rec/(prec+rec) if prec+rec else 0
        rows.append(dict(keyword=kw, tp=tp, fp=fp, fn=fn,
                         precision=prec, recall=rec, f1=f1))
    met_df = pd.DataFrame(rows).set_index("keyword")

    def show_metric(df_subset, metric_name):
        for kw, row in df_subset.head(10).iterrows():
            st.write(f"**{kw}**")
            st.progress(row[metric_name], text=f"{row[metric_name]:.2f}")

    tab_rec, tab_prec, tab_f1 = st.tabs(["✅ Recall", "🎯 Precision", "⚖️ F1"])

    with tab_rec:
        show_metric(met_df.sort_values("recall", ascending=False), "recall")
    with tab_prec:
        show_metric(met_df.sort_values("precision", ascending=False), "precision")
    with tab_f1:
        best = met_df.sort_values("f1", ascending=False)
        show_metric(best, "f1")
        st.markdown("### 🏆 Top‑3 keywords overall")
        for i, (kw, row) in enumerate(best.head(3).iterrows(), 1):
            st.markdown(f"{i}. **{kw}** – F1 {row['f1']:.2f}")

    # --- Example cases ---
    st.header("🔎 Example cases per keyword")
    sel_kw = st.selectbox("Select a keyword", met_df.sort_values("f1", ascending=False).index)

    kw_mask = df["_clean"].str.contains(fr"\b{sel_kw}\b")

    def sample(mask, truth, n=3):
        return df[mask & (df[label_col]==truth)][text_col].head(n).tolist()

    tp_ex = sample( kw_mask, 1)
    fp_ex = sample( kw_mask, 0)
    fn_ex = sample(~kw_mask, 1)

    def show_examples(label, items, color):
        st.markdown(f"**{label}**")
        for t in items:
            st.markdown(f"<span style='background:{color};padding:2px'>{t}</span>", unsafe_allow_html=True)

    show_examples("True positives",  tp_ex, "#d4edda")
    show_examples("False positives", fp_ex, "#f8d7da")
    show_examples("False negatives", fn_ex, "#fff3cd")

    # --- Downloads ---
    st.subheader("💾 Download results")
    st.download_button("classified_results.csv", df.to_csv(index=False).encode(), "classified_results.csv")
    st.download_button("keyword_metrics.csv",   met_df.to_csv().encode(),        "keyword_metrics.csv")
