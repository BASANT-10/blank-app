# classifier_word_metrics.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re, ast

st.set_page_config(page_title="Classifier Word Metrics", layout="wide")
st.title("📈 Classifier Word Metrics")

# ──────────────────────────  STEP 1 – UPLOAD  ──────────────────────────
file = st.file_uploader("📁 **Step 1 – Upload CSV (must include a text column and ground‑truth labels)**",
                        type="csv")

if not file:
    st.info("Awaiting CSV upload…")
    st.stop()

df_raw = pd.read_csv(file)
st.success("Dataset loaded.")
with st.expander("Preview data"):
    st.dataframe(df_raw.head())

# ──────────────────────────  STEP 2 – BASIC SETTINGS  ──────────────────
default_tactics = {
    "urgency_marketing":  ["now", "today", "limited", "hurry", "exclusive"],
    "social_proof":       ["bestseller", "popular", "trending", "recommended"],
    "discount_marketing": ["sale", "discount", "deal", "free", "offer"],
}
tactic = st.selectbox("🎯 **Step 2 – Choose a marketing tactic**", list(default_tactics.keys()))
text_col  = st.selectbox("📝 Text column",  df_raw.columns)
label_col = st.selectbox("🏷️  Ground‑truth label column (1 = positive, 0 = negative)", df_raw.columns)

# ──────────────────────────  helper  ───────────────────────────────────
def clean(txt:str) -> str:
    return re.sub(r"[^A-Za-z0-9\s]", "", str(txt).lower())

# pre‑clean once and store
df = df_raw.copy()
df["_clean"] = df[text_col].astype(str).apply(clean)

# ──────────────────────────  STEP 3 – DICTIONARY SETUP  ────────────────
st.header("🔧 Step 3 – Dictionary setup")

dict_mode = st.radio("Choose one:", ["🧠 Generate from data", "📥 Provide custom dictionary"])
dict_ready = False  # flag

if dict_mode == "🧠 Generate from data":
    # compute top words on demand
    if st.button("Generate dictionary from dataset"):
        word_freq   = pd.Series(" ".join(df["_clean"]).split()).value_counts()
        top_words   = word_freq[word_freq > 1].head(20)
        auto_dict   = {tactic: set(top_words.index)}
        st.write("Top keywords:", top_words)
        st.write("Auto‑generated dictionary:", auto_dict)

        dict_text = st.text_area("✏️ Edit the generated dictionary", value=str(auto_dict), height=150)
        if st.button("Save dictionary"):
            try:
                st.session_state.dictionary = ast.literal_eval(dict_text)
                dict_ready = True
                st.success("Dictionary saved.")
            except Exception:
                st.error("Bad format – please correct and save again.")

elif dict_mode == "📥 Provide custom dictionary":
    custom_example = f"{{'{tactic}': {{'keyword1', 'keyword2'}}}}"
    custom_dict_text = st.text_area("Paste your dictionary (Python syntax)",
                                    value=custom_example,
                                    height=150)
    if st.button("Save custom dictionary"):
        try:
            st.session_state.dictionary = ast.literal_eval(custom_dict_text)
            dict_ready = True
            st.success("Custom dictionary saved.")
        except Exception:
            st.error("Bad format – please correct and save again.")

# propagate flag across reruns
if dict_ready:
    st.session_state.dict_ready = True
if not st.session_state.get("dict_ready"):
    st.info("Dictionary not yet saved – complete Step 3 first.")
    st.stop()

# ──────────────────────────  STEP 4 – EVALUATION  ──────────────────────
st.header("🚀 Step 4 – Run evaluation")
dictionary = st.session_state.dictionary
keywords   = list(dictionary[tactic])

if st.button("Start evaluation"):
    # predictions
    df["_pred"] = df["_clean"].apply(lambda t: 1 if any(k in t.split() for k in keywords) else 0)

    # store keyword matches
    df["_matches"] = df["_clean"].apply(lambda t: [k for k in keywords if k in t.split()])

    # ----------------------  RESULTS (TRUE POSITIVES) -------------------
    st.subheader("📝 Classification results – **True positives**")
    tp_df = df[(df[label_col] == 1) & (df["_pred"] == 1)]
    show_all = st.checkbox("Show all true‑positive rows", value=False)
    st.dataframe(tp_df.head(20) if not show_all else tp_df)

    accuracy = (df[label_col] == df["_pred"]).mean()
    st.metric("Overall accuracy", f"{accuracy*100:.1f} %")

    # ----------------------  KEYWORD IMPACT ANALYSIS --------------------
    st.header("📊 Keyword impact analysis")

    # confusion components per keyword
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

    # --- TABS: recall / precision / f1
    tab_recall, tab_prec, tab_f1 = st.tabs(["✅ Recall", "🎯 Precision", "⚖️ F1"])

    def show_progress(df, metric, color):
        for kw, row in df.head(10).iterrows():
            st.write(f"**{kw}**")
            st.progress(row[metric], text=f"{row[metric]:.2f}")

    with tab_recall:
        show_progress(met_df.sort_values("recall", ascending=False), "recall", "green")
    with tab_prec:
        show_progress(met_df.sort_values("precision", ascending=False), "precision", "blue")
    with tab_f1:
        best = met_df.sort_values("f1", ascending=False)
        show_progress(best, "f1", "purple")
        st.markdown("### 🏆 Top‑3 overall keywords")
        for i,(kw,row) in enumerate(best.head(3).iterrows(),1):
            st.markdown(f"**{i}. {kw}** – F1 {row['f1']:.2f}")

    # ----------------------  EXAMPLE CASES  -----------------------------
    st.header("🔎 Example cases per keyword")
    sel_kw = st.selectbox("Choose a keyword", met_df.sort_values("f1", ascending=False).index)

    mask_kw  = df["_clean"].str.contains(fr"\b{sel_kw}\b")

    def sample(mask, truth, n=3):
        return df[mask & (df[label_col]==truth)][text_col].head(n).tolist()

    tp_ex = sample(mask_kw, 1)
    fp_ex = sample(mask_kw, 0)
    fn_ex = sample(~mask_kw, 1)

    def highlight(lst, col):
        for t in lst:
            st.markdown(f"<span style='background:{col};padding:2px'>{t}</span>",
                        unsafe_allow_html=True)

    st.markdown("#### ✅ True positives")
    highlight(tp_ex, "#d4edda")
    st.markdown("#### ❌ False positives")
    highlight(fp_ex, "#f8d7da")
    st.markdown("#### ⚠️ False negatives")
    highlight(fn_ex, "#fff3cd")
