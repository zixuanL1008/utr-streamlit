import os, subprocess
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

@st.cache_resource(show_spinner=False)
def load_insight_model():
    from insight import ConvTransformerPredictor, alphabet, device
    model = ConvTransformerPredictor(alphabet).to("cpu")
    weights = os.getenv("INSIGHT_WEIGHTS", str(BASE_DIR / "models" / "esm_weights.pkl"))
    if not Path(weights).exists():
        return model, False
    sd = torch.load(weights, map_location="cpu")
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.esm2.load_state_dict(sd)
    model.eval()
    return model, True

@st.cache_resource(show_spinner=False)
def load_vienna():
    try:
        import RNA
        return RNA
    except Exception:
        return None

def compute_mfe(seq, RNA):
    if RNA is None: return np.nan
    rna = seq.replace("T", "U")
    _, mfe = RNA.fold(rna)
    return float(mfe)

st.set_page_config(page_title="UTR Generator (Streamlit)", layout="centered")
st.title("🧬 UTR Generator")

with st.sidebar:
    gene   = st.text_input("Gene", value="TP53")
    length = st.number_input("UTR length", 10, 512, 80, 10)
    num    = st.number_input("数量 / batch size", 1, 1000, 10, 1)
    gc     = st.slider("目标 GC（0~1）", 0.0, 1.0, 0.6, 0.05)
    st.caption("未放权重也可运行，Insight 分数会为空。")

if st.button("🚀 Start Generation", use_container_width=True):
    if not gene.strip():
        st.error("Gene 不能为空"); st.stop()

    progress = st.progress(0, text="准备中…")

    # 1) 调用 single-gene-nb.py 生成序列（注意 -gc 需要负号）
    progress.progress(10, text="生成序列…")
    gc_pct = int(round(gc * 100))
    cmd = ["python", str(BASE_DIR / "single-gene-nb.py"),
           "-g", gene, "-s", "500", "-bs", str(int(num)), "-gc", f"-{gc_pct}"]
    try:
        subprocess.run(cmd, check=True, cwd=str(BASE_DIR))
    except subprocess.CalledProcessError as e:
        st.error(f"生成序列失败：{e}"); st.stop()

    # 兼容两种可能命名
    candidates = [OUTPUT_DIR / f"{gene}_best_seqs.txt", OUTPUT_DIR / f"best_seqs_{gene}.txt"]
    seq_file = next((p for p in candidates if p.exists()), None)
    if not seq_file:
        st.error("未找到序列文件，请检查 single-gene-nb.py 的输出命名。"); st.stop()

    seqs = [s.strip() for s in seq_file.read_text().splitlines() if s.strip()]
    control_seq = "GAGAATAAACTAGTATTCTTCTGGTCCCCACAGACTCAGAGAGAACCCGCCACC"
    seqs.append(control_seq)

    # 2) Insight 打分
    progress.progress(55, text="加载 Insight 模型…")
    model, ok = load_insight_model()
    scores = []
    if ok:
        from insight import fitness_function
        for i, s in enumerate(seqs, 1):
            try: scores.append(float(fitness_function(s, model, "cpu")))
            except Exception: scores.append(np.nan)
            if i % 5 == 0: progress.progress(min(90, 60 + int(30 * i/len(seqs))))
    else:
        scores = [np.nan] * len(seqs)
        st.info("未加载权重，Insight 分数为空。")

    # 3) MFE
    RNA = load_vienna()
    progress.progress(95, text="计算 MFE…")
    mfes = [compute_mfe(s, RNA) for s in seqs]

    # 4) 展示 + 下载
    progress.progress(100, text="完成 ✅")
    df = pd.DataFrame({"sequence": seqs, "insight_score": scores, "mfe_kcal_mol": mfes})
    st.subheader("Best Sequence（按 Insight 分数）")
    view = df.sort_values("insight_score", ascending=False, na_position="last").reset_index(drop=True)
    if not view.empty: st.code(view.iloc[0]["sequence"])
    st.subheader("全部结果")
    st.dataframe(view, use_container_width=True, height=420)
    st.download_button("📥 下载 CSV", view.to_csv(index=False).encode("utf-8"),
                       file_name=f"{gene}_final.csv", mime="text/csv", use_container_width=True)
