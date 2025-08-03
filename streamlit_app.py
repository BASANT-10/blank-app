# classifier_word_metrics.py  –  streamlit run classifier_word_metrics.py
import streamlit as st
import pandas as pd
import re, ast

st.set_page_config(page_title="Classifier Word Metrics", layout="wide")
st.title("📈 Classifier Word Metrics")

# ───────────────────────── STEP 1 – UPLOAD DATA ─────────────────────────
file = st.file_uploader("📁 **Step 1 – Upload CSV**", type="csv")
if not file:
    st.info("Upload a CSV to continue."); st.stop()

df_raw = pd.read_csv(file)
st.success("Dataset loaded.")
with st.expander("Preview data"):
    st.dataframe(df_raw.head())

# ───────────────────────── STEP 2 – BASIC SETTINGS ──────────────────────
default_tactics = {
    "urgency_marketing":  ["now","today","limited","hurry","exclusive"],
    "social_proof":       ["bestseller","popular","trending","recommended"],
    "discount_marketing": ["sale","discount","deal","free","offer"],
}
tactic = st.selectbox("🎯 Tactic to evaluate", list(default_tactics.keys()))
text_col  = st.selectbox("📝 Text column",  df_raw.columns)
label_col = st.selectbox("🏷️ Ground‑truth 0/1 column", df_raw.columns)

# optional ID
id_options = ["<none – keep each row>"] + list(df_raw.columns)
id_col = st.selectbox("🆔 (Optional) ID column for aggregation", id_options)

# clean text once
clean = lambda t: re.sub(r"[^A-Za-z0-9\s]", "", str(t).lower())
df = df_raw.copy()
df["_clean"] = df[text_col].astype(str).apply(clean)

# ───────────────────────── STEP 3 – DICTIONARY SETUP ────────────────────
st.header("🔧 Step 3 – Dictionary setup")
dict_mode = st.radio("Choose method:", ["🧠 Generate from data","📥 Provide custom dictionary"])

# reset session keys if switching modes
if "last_mode" in st.session_state and st.session_state.last_mode != dict_mode:
    for k in ("gen_ready","dict_ready","dictionary","auto_dict","top_words"):
        st.session_state.pop(k, None)
st.session_state.last_mode = dict_mode

# ----- Option A: generate from data -----
if dict_mode == "🧠 Generate from data":
    if st.button("Generate top‑20 keyword dictionary"):
        freq = pd.Series(" ".join(df["_clean"]).split()).value_counts()
        top  = freq[freq>1].head(20)
        st.session_state.top_words = top
        st.session_state.auto_dict = {tactic:set(top.index)}
        st.session_state.gen_ready = True

    if st.session_state.get("gen_ready"):
        st.dataframe(st.session_state.top_words)
        dict_text = st.text_area("Edit dictionary",
                                 value=str(st.session_state.auto_dict),
                                 height=150, key="dict_edit")
        if st.button("Save dictionary"):
            try:
                st.session_state.dictionary = ast.literal_eval(dict_text)
                st.session_state.dict_ready = True
                st.success("Dictionary saved.")
            except:
                st.error("Bad format – fix and save again.")

# ----- Option B: paste custom dictionary -----
else:
    template = f"{{'{tactic}': {{'keyword1','keyword2'}}}}"
    dict_text = st.text_area("Paste dictionary", value=template, height=150)
    if st.button("Save custom dictionary"):
        try:
            st.session_state.dictionary = ast.literal_eval(dict_text)
            st.session_state.dict_ready = True
            st.success("Dictionary saved.")
        except:
            st.error("Bad format – fix and save again.")

if not st.session_state.get("dict_ready"):
    st.info("Save a dictionary to unlock evaluation."); st.stop()

dictionary = st.session_state.dictionary
keywords   = list(dictionary[tactic])

# ───────────────────────── STEP 4 – EVALUATION ─────────────────────────
st.header("🚀 Step 4 – Run evaluation")

if st.button("Start evaluation"):
    # row‑level keyword matches & prediction
    df["_matches"] = df["_clean"].apply(lambda t:[k for k in keywords if k in t.split()])
    df["_pred"]    = df["_matches"].apply(lambda m: 1 if m else 0)
    df["_coverage"]= df["_clean"].apply(lambda t: sum(1 for w in t.split() if w in keywords)/len(t.split()) if t.split() else 0)

    # ───────────── token‑level long table (NEW) ─────────────
    token_rows = []
    for idx, row in df.iterrows():
        tokens = row["_clean"].split()
        for tok in tokens:
            token_rows.append({
                "row_index": idx,
                **({id_col: row[id_col]} if id_col != "<none – keep each row>" else {}),
                "token": tok,
                "is_keyword": 1 if tok in keywords else 0
            })
    token_df = pd.DataFrame(token_rows)

    st.subheader("🔠 Token‑level tagging (first 100 tokens)")
    st.dataframe(token_df.head(100))

    # ───────────── row‑level true positives  ─────────────
    st.subheader("📝 Row‑level true positives (first 20)")
    tp_rows = df[(df[label_col]==1)&(df["_pred"]==1)]
    st.dataframe(tp_rows.head(20))

    row_acc = (df[label_col]==df["_pred"]).mean()
    st.metric("Row‑level accuracy", f"{row_acc*100:.1f}%")

    # ───────────── ID‑level aggregation (optional) ──────────
    if id_col != "<none – keep each row>":
        st.subheader("🆔 ID‑level aggregation")
        grp = df.groupby(id_col)
        id_df = pd.DataFrame({
            "true_label": grp[label_col].max(),
            "pred_label": grp["_pred"].max(),
            "avg_coverage": grp["_coverage"].mean()
        }).reset_index()
        id_acc = (id_df["true_label"]==id_df["pred_label"]).mean()
        st.metric("ID‑level accuracy", f"{id_acc*100:.1f}%")
        st.dataframe(id_df.head(20))

    # ───────────── keyword metrics  ─────────────
    st.header("📊 Keyword impact analysis")
    rows=[]
    for kw in keywords:
        m = df["_clean"].str.contains(fr"\b{kw}\b")
        tp=((m)&(df[label_col]==1)).sum()
        fp=((m)&(df[label_col]==0)).sum()
        fn=((~m)&(df[label_col]==1)).sum()
        prec=tp/(tp+fp) if tp+fp else 0
        rec =tp/(tp+fn) if tp+fn else 0
        f1  =2*prec*rec/(prec+rec) if prec+rec else 0
        rows.append(dict(keyword=kw,tp=tp,fp=fp,fn=fn,precision=prec,recall=rec,f1=f1))
    met_df = pd.DataFrame(rows).set_index("keyword")

    tabs = st.tabs(["Recall","Precision","F1"])
    for tab,metric in zip(tabs,["recall","precision","f1"]):
        with tab:
            for kw,row in met_df.sort_values(metric,ascending=False).head(10).iterrows():
                st.write(f"**{kw}**")
                st.progress(row[metric], text=f"{row[metric]:.2f}")

    # ───────────── example sentences  ─────────────
    st.header("🔎 Example cases")
    sel_kw = st.selectbox("Keyword", met_df.sort_values("f1",ascending=False).index)
    kw_mask = df["_clean"].str.contains(fr"\b{sel_kw}\b")
    def sample(mask,truth): return df[mask & (df[label_col]==truth)][text_col].head(3).tolist()
    TP,FP,FN = sample( kw_mask,1), sample( kw_mask,0), sample(~kw_mask,1)
    def show(tag,lst,col):
        st.markdown(f"**{tag}**")
        for t in lst:
            st.markdown(f"<span style='background:{col};padding:2px'>{t}</span>", unsafe_allow_html=True)
    show("True positives",TP,"#d4edda"); show("False positives",FP,"#f8d7da"); show("False negatives",FN,"#fff3cd")

    # ───────────── downloads  ─────────────
    st.subheader("💾 Download results")
    st.download_button("rows_classified.csv",  df.to_csv(index=False).encode(),  "rows_classified.csv")
    st.download_button("keyword_metrics.csv", met_df.to_csv().encode(),         "keyword_metrics.csv")
    st.download_button("token_tags.csv",      token_df.to_csv(index=False).encode(), "token_tags.csv")
    if id_col != "<none – keep each row>":
        st.download_button("ids_classified.csv", id_df.to_csv(index=False).encode(), "ids_classified.csv")
