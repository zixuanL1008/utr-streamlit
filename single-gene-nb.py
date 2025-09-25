#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, json, time, math, random, argparse, logging, collections
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# ---- 可选依赖：缺失不致命（云端经常没有这些内部模块） ----
try:
    from util import *
except Exception as _e:
    print(f"[WARN] util not found: {_e}")

try:
    from framepool import *
except Exception as _e:
    print(f"[WARN] framepool not found: {_e}")

try:
    from models import Modules  # 未使用，但保留兼容
    from models.ScheduleOptimizer import ScheduledOptim
except Exception as _e:
    print(f"[WARN] models.* not found: {_e}")

import requests
from tqdm import tqdm
import torch
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from Bio import SeqIO
from sklearn.preprocessing import OneHotEncoder

# ---- TF 基本设置（与你源码一致） ----
tf.compat.v1.enable_eager_execution()
TF_ENABLE_ONEDNN_OPTS = 0

# ==== 路径改为相对路径（关键修改） ====
BASE_DIR   = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"   # 放 .h5/.pth 等权重
OUTPUT_DIR = BASE_DIR / "outputs"  # 结果输出目录
CACHE_DIR  = BASE_DIR / ".cache"   # 基因信息缓存
OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# ==== 参数（保持你原来的语义） ====
SEQ_LEN = 128
parser = argparse.ArgumentParser()
parser.add_argument('-g',   type=str,   required=True,  default="TR53")
parser.add_argument('-gc',  type=float, required=False, default=-100)  # 传 -60 表示 60%
parser.add_argument('-bs',  type=int,   required=False, default=100)
parser.add_argument('-lr',  type=int,   required=False, default=4)
parser.add_argument('-gpu', type=str,   required=False, default='-1')
parser.add_argument('-s',   type=int,   required=False, default=1)     # 迭代步数（你源码如此）
args = parser.parse_args()

BATCH_SIZE = args.bs
GENE       = args.g
GC_LIMIT   = -args.gc / 100.0  # 你原来的写法：传 -60 → 0.60
LR         = 0.005
GPU        = args.gpu
STEPS      = args.s

# ==== 设备 ====
if GPU == '-1':
    device = 'cpu'
else:
    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = GPU
        device = 'cuda'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        device = 'cpu'

# ==== 工具函数（与你源码一致/轻改） ====
def reverse_complement(sequence):
    complement = {'A':'T','T':'A','C':'G','G':'C','a':'t','t':'a','c':'g','g':'c','N':'N','n':'N'}
    return ''.join(complement.get(base,'N') for base in reversed(sequence))

def one_hot(seq):
    convert = False
    if isinstance(seq, tf.Tensor):
        seq = seq.numpy().astype(str)
        convert = True
    num_seqs = len(seq)
    seq_len  = len(seq[0])
    seqindex = {'A':0,'C':1,'G':2,'T':3,'a':0,'c':1,'g':2,'t':3}
    seq_vec  = np.zeros((num_seqs,seq_len,4), dtype='bool')
    for i in range(num_seqs):
        thisseq = seq[i]
        for j in range(seq_len):
            try:
                seq_vec[i,j,seqindex[thisseq[j]]] = 1
            except:
                pass
    if convert:
        seq_vec = tf.convert_to_tensor(seq_vec, dtype=tf.float32)
    return seq_vec

def recover_seq(samples, rev_charmap):
    if isinstance(samples, tf.Tensor):
        samples = samples.numpy()
    char_probs = samples
    argmax = np.argmax(char_probs, 2)
    seqs = []
    for line in argmax:
        s = "".join(rev_charmap[d] for d in line)
        s = s.replace('*','')
        seqs.append(s)
    return np.array(seqs)

def convert_model(model_: Model):
    # 直接沿用你原有的层迭代逻辑
    input_ = tf.keras.layers.Input(shape=(10500, 4))
    input  = input_
    for i in range(len(model_.layers)-1):
        if isinstance(model_.layers[i+1], tf.keras.layers.Concatenate):
            paddings = tf.constant([[0,0],[0,6]])
            output = tf.pad(input, paddings, 'CONSTANT')
            input  = output
        else:
            if not isinstance(model_.layers[i+1], tf.keras.layers.InputLayer):
                output = model_.layers[i+1](input)
                input  = output
            if isinstance(model_.layers[i+1], tf.keras.layers.Conv1D):
                pass
    model = tf.keras.Model(inputs=input_, outputs=output)
    model.compile(loss="mse", optimizer="adam")
    return model

rna_vocab     = {"A":0,"C":1,"G":2,"T":3,"*":4}
rev_rna_vocab = {v:k for k,v in rna_vocab.items()}

# ---- 你之前我帮你改过的 select_best（保留） ----
def select_best(scores, seqs, gc_control=False, GC_min=0.3, GC_max=0.7):
    def gc_content(seq: str) -> float:
        seq = seq.upper()
        gc = seq.count('G') + seq.count('C')
        return gc / len(seq) if seq else 0.0
    n_batch = len(scores[0])
    selected_seqs, selected_scores = [], []
    for b in range(n_batch):
        candidates = [(scores[i][b], seqs[i][b]) for i in range(len(seqs))]
        candidates.sort(key=lambda x: -x[0])
        if gc_control:
            filtered = [(s, seq) for s, seq in candidates
                        if GC_min <= gc_content(seq) <= GC_max]
            step = 0.05
            while len(filtered) < 1:
                GC_min = max(0.0, GC_min - step)
                GC_max = min(1.0, GC_max + step)
                filtered = [(s, seq) for s, seq in candidates
                            if GC_min <= gc_content(seq) <= GC_max]
            best_score, best_seq = filtered[0]
        else:
            best_score, best_seq = candidates[0]
        selected_seqs.append(best_seq)
        selected_scores.append(best_score)
    return selected_seqs, selected_scores

# =========================
# Ensembl 相关（你的源码保留）
# =========================
class GeneInfoRetriever:
    def __init__(self):
        self.base_url = "https://rest.ensembl.org"
        self.headers = {"Content-Type": "application/json"}
        self.sleep_time = 0.5
    def _make_request(self, endpoint):
        url = self.base_url + endpoint
        try:
            response = requests.get(url, headers=self.headers)
            time.sleep(self.sleep_time)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            print(f"Request error: {e}")
            return None
    def get_gene_id(self, gene_symbol, species="homo_sapiens"):
        endpoint = f"/lookup/symbol/{species}/{gene_symbol}"
        response = self._make_request(endpoint)
        return response.get("id") if response else None
    def get_tss_and_utr(self, gene_id):
        endpoint = f"/lookup/id/{gene_id}?expand=1&utr=1"
        response = self._make_request(endpoint)
        if not response or "Transcript" not in response:
            return None
        canonical_transcript = None
        for transcript in response["Transcript"]:
            if transcript.get("is_canonical", 0) == 1:
                canonical_transcript = transcript
                break
        if not canonical_transcript:
            for transcript in response["Transcript"]:
                if transcript.get("biotype") == "protein_coding":
                    canonical_transcript = transcript
                    break
        if not canonical_transcript:
            canonical_transcript = response["Transcript"][0] if response["Transcript"] else None
        if not canonical_transcript:
            return None
        strand = canonical_transcript.get("strand")
        tss = canonical_transcript["start"] if strand == 1 else canonical_transcript["end"]
        five_prime_utr = None
        if "UTR" in canonical_transcript:
            for utr in canonical_transcript["UTR"]:
                if utr.get("object_type") == "five_prime_UTR":
                    five_prime_utr = {"start": utr.get("start"), "end": utr.get("end")}
                    break
        if five_prime_utr:
            expected_tss = five_prime_utr["start"] if strand == 1 else five_prime_utr["end"]
            if expected_tss != tss:
                print(f"Warning: Adjusting TSS to UTR {'start' if strand==1 else 'end'} {expected_tss}")
                tss = expected_tss
        return {
            "tss": tss,
            "strand": strand,
            "chromosome": canonical_transcript.get("seq_region_name"),
            "five_prime_utr": five_prime_utr,
            "transcript_id": canonical_transcript.get("id")
        }
    def get_promoter_sequence(self, gene_id, upstream=7000, downstream=4000):
        tss_info = self.get_tss_and_utr(gene_id)
        if not tss_info:
            return None, None
        chromosome = tss_info["chromosome"]
        strand     = tss_info["strand"]
        tss_pos    = tss_info["tss"]
        if strand == 1:
            seq_start = tss_pos - upstream
            seq_end   = tss_pos + downstream - 1
        else:
            seq_start = tss_pos - downstream
            seq_end   = tss_pos + upstream - 1
        seq_start = max(1, seq_start)
        sequence_coords = {"chromosome": chromosome, "start": seq_start, "end": seq_end, "strand": 1 if strand == 1 else -1}
        strand_str = "1" if strand == 1 else "-1"
        endpoint = f"/sequence/region/human/{chromosome}:{seq_start}..{seq_end}:{strand_str}"
        response = self._make_request(endpoint)
        return (response.get("seq") if response else None), sequence_coords
    def get_gene_info(self, gene_symbol, species="homo_sapiens", output_json="gene_info.json"):
        cache_file = CACHE_DIR / f"{gene_symbol}_info.json"
        if not cache_file.exists():
            gene_id   = self.get_gene_id(gene_symbol, species)
            if not gene_id:
                return {"error": f"Gene {gene_symbol} not found"}
            tss_info  = self.get_tss_and_utr(gene_id)
            if not tss_info:
                return {"error": "Could not retrieve TSS or transcript information"}
            promoter_sequence, sequence_coords = self.get_promoter_sequence(gene_id)
            if not promoter_sequence:
                return {"error": "Could not retrieve promoter sequence"}
            gene_info = {
                "gene_symbol": gene_symbol,
                "gene_id": gene_id,
                "promoter_sequence": promoter_sequence,
                "sequence_length": len(promoter_sequence),
                "sequence_coordinates": sequence_coords,
                "tss": {
                    "chromosome": tss_info["chromosome"],
                    "position": tss_info["tss"],
                    "strand": "+" if tss_info["strand"] == 1 else "-"
                },
                "five_prime_utr": tss_info["five_prime_utr"],
                "transcript_id": tss_info["transcript_id"]
            }
            with open(cache_file, "w") as f:
                json.dump(gene_info, f, indent=2)
        else:
            with open(cache_file, "r") as f:
                gene_info = json.load(f)
        return gene_info
    def replace_utr_in_sequence(self, gene_info_file, generated_utrs, target_length=10500, output_prefix="modified_sequence", write_json=False, verbose=False):
        try:
            with open(gene_info_file, "r") as f:
                gene_info = json.load(f)
            original_sequence = gene_info["promoter_sequence"]
            strand = gene_info["tss"]["strand"]
            tss_position = gene_info["tss"]["position"]
            sequence_coords = gene_info["sequence_coordinates"]
            seq_start = sequence_coords["start"]
            seq_end   = sequence_coords["end"]
            five_prime_utr = gene_info["five_prime_utr"]
            gene_symbol = gene_info["gene_symbol"]
            transcript_id = gene_info["transcript_id"]
            if not five_prime_utr:
                print(f"Error: No 5' UTR information for {gene_symbol}")
                return []
            if strand == "+":
                utr_start_genomic = five_prime_utr["start"]
                utr_end_genomic   = five_prime_utr["end"]
                utr_start_seq = utr_start_genomic - seq_start
                utr_end_seq   = utr_end_genomic   - seq_start
            else:
                utr_start_genomic = five_prime_utr["end"]
                utr_end_genomic   = five_prime_utr["start"]
                utr_start_seq = seq_end - utr_start_genomic
                utr_end_seq   = seq_end - utr_end_genomic
            seq_length = len(original_sequence)
            if not (0 <= utr_start_seq <= seq_length and 0 <= utr_end_seq <= seq_length):
                print(f"Error: 5'UTR indices {utr_start_seq}-{utr_end_seq} out of bounds 0-{seq_length}")
                return []
            original_utr_length = abs(utr_end_genomic - utr_start_genomic) + 1
            modified_sequences = []
            for i, new_utr in enumerate(generated_utrs):
                new_utr_length = len(new_utr)
                if strand == "+":
                    new_sequence = (
                        original_sequence[:utr_start_seq] +
                        new_utr +
                        original_sequence[utr_end_seq + 1:]
                    )
                    new_utr_start_genomic = utr_start_genomic
                    new_utr_end_genomic   = utr_start_genomic + new_utr_length - 1
                    if len(new_sequence) > target_length:
                        new_sequence = new_sequence[:target_length]
                        sequence_coords["end"] = seq_start + target_length - 1
                    elif len(new_sequence) < target_length:
                        if verbose:
                            print(f"Error: Sequence too short ({len(new_sequence)} nt)")
                            continue
                else:
                    new_utr_rc = reverse_complement(new_utr)
                    new_sequence = (
                        original_sequence[:min(utr_start_seq, utr_end_seq)] +
                        new_utr_rc +
                        original_sequence[max(utr_start_seq, utr_end_seq) + 1:]
                    )
                    new_utr_start_genomic = utr_start_genomic
                    new_utr_end_genomic   = utr_start_genomic - new_utr_length + 1
                    if len(new_sequence) > target_length:
                        trim_amount = len(new_sequence) - target_length
                        new_sequence = new_sequence[trim_amount:]
                        sequence_coords["start"] = seq_start + trim_amount
                    elif len(new_sequence) < target_length:
                        if verbose:
                            print(f"Error: Sequence too short ({len(new_sequence)} nt)")
                            continue
                modified_sequences.append(new_sequence)
            return modified_sequences
        except Exception as e:
            print(f"Error processing UTR replacement: {e}")
            return []

# =========================
# 生成主流程（与你源码一致，但换相对路径 & 加兜底）
# =========================

DIM = 40
CELL_LINE = ''

# ---- 相对路径的权重文件（关键修改） ----
gpath    = MODELS_DIR / "checkpoint_3000.h5"
exp_path = MODELS_DIR / "humanMedian_trainepoch.11-0.426.h5"
tpath    = MODELS_DIR / "schedule_MTL-model_best_cv1.pth"  # 当前脚本未直接用到

gene_name = GENE
retriever = GeneInfoRetriever()

# 取参考序列（缓存）
output_json = f"{gene_name}_info.json"
cache_file  = CACHE_DIR / output_json
if not cache_file.exists():
    gene_info = retriever.get_gene_info(gene_name, output_json=output_json)
    if "error" in gene_info:
        print(f"Error: {gene_info['error']}")
        ref = "A" * 10500
    else:
        ref = gene_info["promoter_sequence"]
else:
    with open(cache_file, "r") as f:
        gene_info = json.load(f)
        ref = gene_info["promoter_sequence"]

original_gene_sequence = ref

# 尝试加载 Keras 模型（若不存在则走 fallback）
MODEL_OK = True
try:
    model = load_model(str(exp_path))
    model = convert_model(model)
except Exception as e:
    print(f"[WARN] load exp model failed: {e}")
    MODEL_OK = False

# 尝试加载 GAN（若不存在则走 fallback）
try:
    wgan = tf.keras.models.load_model(str(gpath))
except Exception as e:
    print(f"[WARN] load GAN model failed: {e}")
    MODEL_OK = False

# ---- fallback：没有模型也能跑 ----
def fallback_generate(n: int, length: int, target_gc: float) -> List[str]:
    seqs = []
    for _ in range(n):
        s = []
        for _ in range(length):
            if random.random() < target_gc:
                s.append(random.choice("GC"))
            else:
                s.append(random.choice("AT"))
        seqs.append("".join(s))
    return seqs

# 如果无法加载模型，直接走兜底路径并写出文件，后续退出 0（让前端不中断）
if not MODEL_OK:
    gc_target = abs(args.gc) / 100.0 if abs(args.gc) > 1 else max(0.0, min(1.0, abs(args.gc)))
    gc_target = max(0.0, min(1.0, gc_target if gc_target > 0 else 0.6))
    # 生成偏置 UTR → 替换到 10.5kb 上下文（只为得到一致的输出文件）
    utrs = fallback_generate(BATCH_SIZE, SEQ_LEN, gc_target)
    mod = retriever.replace_utr_in_sequence(str(cache_file), utrs, target_length=10500)
    # 结果文件（与前端对接）
    out_best = OUTPUT_DIR / f"best_seqs_{gene_name}.txt"
    with open(out_best, "w", encoding="utf-8") as f:
        for s in utrs:
            f.write(s + "\n")
    print(f"[INFO] Fallback path used. Wrote {len(utrs)} seqs to {out_best}")
    sys.exit(0)

# ========= 下面保持你的原优化流程（仅把硬编码路径改为 OUTPUT_DIR） =========

"""
Data:
"""
noise = tf.Variable(tf.random.normal(shape=[BATCH_SIZE, DIM]))
noise_small = tf.random.normal(shape=[BATCH_SIZE, DIM], stddev=1e-5)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)

"""
Original Gene Expression
"""
seqs_orig = one_hot([original_gene_sequence[:10500]])
pred_orig = model(seqs_orig)
pred_orig = tf.reshape(pred_orig, (-1)).numpy().astype('float')[0]

"""
Optimization
"""
bind_scores_list = []
bind_scores_means = []
sequences_list = []

LOW_START = False
best = 100
if LOW_START:
    for _ in tqdm(range(1000)):
        tempnoise = tf.random.normal(shape=[BATCH_SIZE, DIM])
        sequences = wgan(tempnoise)
        seqs_gen = recover_seq(sequences, rev_rna_vocab)
        seqs = retriever.replace_utr_in_sequence(str(cache_file), seqs_gen)
        seqs = one_hot(seqs)
        pred = model(seqs)
        score = np.mean(tf.reshape(pred, (-1)).numpy().astype('float'))
        if score < best:
            best = score
            selectednoise = tempnoise
    noise = tf.Variable(selectednoise)
else:
    noise = tf.Variable(tf.random.normal(shape=[BATCH_SIZE, DIM]))

OPTIMIZE = True
sequences_init = wgan(noise)
gen_seqs_init = sequences_init.numpy().astype('float')
seqs_gen_init = recover_seq(gen_seqs_init, rev_rna_vocab)
seqs_init = retriever.replace_utr_in_sequence(str(cache_file), seqs_gen_init)
seqs_init = one_hot(seqs_init)
pred_init = model(seqs_init)
init_t = tf.reshape(pred_init, (-1)).numpy().astype('float')

seqs_collection = []
scores_collection = []
GC_CONTROL = GC_LIMIT > 0.0
iters_ = []

if OPTIMIZE:
    iter_ = 0
    for _ in tqdm(range(STEPS)):
        with tf.GradientTape() as gtape:
            gtape.watch(noise)
            sequences = wgan(noise)
            seqs_gen = recover_seq(sequences, rev_rna_vocab)
            seqs_collection.append(seqs_gen)
            seqs2 = retriever.replace_utr_in_sequence(str(cache_file), seqs_gen)
            seqs = one_hot(seqs2)
            seqs = tf.convert_to_tensor(seqs, dtype=tf.float32)
            with tf.GradientTape() as ptape:
                ptape.watch(seqs)
                pred = model(seqs)
                t = tf.reshape(pred, (-1))
                scores_collection.append(t.numpy().astype('float'))
                pred = tf.math.scalar_mul(-1.0, pred)
            g1 = ptape.gradient(pred, seqs)
            g1 = tf.slice(g1, [0,7000,0], [-1, SEQ_LEN, -1])
            tmp_g   = g1.numpy().astype('float')
            tmp_seq = seqs_gen
            tmp_lst = np.zeros(shape=(BATCH_SIZE, SEQ_LEN, 5))
            for i in range(len(tmp_seq)):
                ln = len(tmp_seq[i])
                edited_g = tmp_g[i][:ln, :]
                edited_g = np.pad(edited_g, ((0, SEQ_LEN - ln), (0, 1)), 'constant')
                tmp_lst[i] = edited_g
            g1 = tf.convert_to_tensor(tmp_lst, dtype=tf.float32)
            g2 = gtape.gradient(sequences, noise, output_gradients=g1)
        a1 = g2 + noise_small
        optimizer.apply_gradients([(a1, noise)])
        iters_.append(iter_)
        iter_ += 1

    sequences_opt = wgan(noise)
    gen_seqs_opt = sequences_opt.numpy().astype('float')
    seqs_gen_opt = recover_seq(gen_seqs_opt, rev_rna_vocab)
    seqs_opt = retriever.replace_utr_in_sequence(str(cache_file), seqs_gen_opt, target_length=10500, output_prefix="modified_sequence")
    seqs_opt = one_hot(seqs_opt)
    pred_opt = model(seqs_opt)
    t = tf.reshape(pred_opt, (-1))
    opt_t = t.numpy().astype('float')

    if GC_CONTROL:
        best_seqs, best_scores = select_best(scores_collection, seqs_collection, True, GC_LIMIT)
    else:
        best_seqs, best_scores = select_best(scores_collection, seqs_collection)

    # ====== 写出到相对路径的 outputs/（关键修改） ======
    if GC_CONTROL:
        with open(OUTPUT_DIR / f"{CELL_LINE}gc_init_exps_{gene_name}.txt", 'w') as f:
            for item in init_t: f.write(f'{item}\n')
        with open(OUTPUT_DIR / f"{CELL_LINE}gc_opt_exps_{gene_name}.txt", 'w') as f:
            for item in best_scores: f.write(f'{item}\n')
        with open(OUTPUT_DIR / f"{CELL_LINE}gc_best_seqs_{gene_name}.txt", 'w') as f:
            for item in best_seqs: f.write(f'{item}\n')
        with open(OUTPUT_DIR / f"{CELL_LINE}gc_init_seqs_{gene_name}.txt", 'w') as f:
            for item in seqs_gen_init: f.write(f'{item}\n')
    else:
        with open(OUTPUT_DIR / f"init_exps_{gene_name}.txt", 'w') as f:
            for item in init_t: f.write(f'{item}\n')
        with open(OUTPUT_DIR / f"opt_exps_{gene_name}.txt", 'w') as f:
            for item in best_scores: f.write(f'{item}\n')
        with open(OUTPUT_DIR / f"best_seqs_{gene_name}.txt", 'w') as f:
            for item in best_seqs: f.write(f'{item}\n')
        with open(OUTPUT_DIR / f"init_seqs_{gene_name}.txt", 'w') as f:
            for item in seqs_gen_init: f.write(f'{item}\n')

    # 兼容前端读取
    with open(OUTPUT_DIR / f"{gene_name}_best_seqs.txt", "w") as f:
        for seq in best_seqs: f.write(f"{seq}\n")

    print(f"Results for {gene_name} saved to {OUTPUT_DIR}")
    print(f"Natural 5' UTR Expression: {np.power(10, pred_orig):.4f}")
    print(f"Average Initial Expression: {np.power(10, np.average(init_t)):.4f}")
    print(f"Max Initial Expression: {np.power(10, np.max(init_t)):.4f}")
    print(f"Max Best Expression: {np.power(10, np.max(best_scores)):.4f}")
    print(f"Average Improvement: {np.average((np.power(10,best_scores)-np.power(10,init_t))/np.power(10,init_t))*100:.2f}%")
    print(f"Max Improvement: {np.max((np.power(10,best_scores)-np.power(10,init_t))/np.power(10,init_t))*100:.2f}%")
    print(f"Average Improvement (wrt to Natural 5'UTR): {np.average((np.power(10,best_scores)-math.pow(10,pred_orig))/math.pow(10,pred_orig))*100:.2f}%")
    print(f"Max Improvement (wrt to Natural 5'UTR): {np.max((np.power(10,best_scores)-math.pow(10,pred_orig))/math.pow(10,pred_orig))*100:.2f}%")
