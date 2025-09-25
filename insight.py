import os
import argparse
from argparse import Namespace
import pathlib
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import esm
from esm.data import *
from esm.model.esm2_secondarystructure import ESM2 as ESM2_SISS
from esm.model.esm2_supervised import ESM2
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer
from esm.modules import ConvTransformerLayer

import numpy as np
import pandas as pd
import random
import math
import scipy.stats as stats
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import r2_score, f1_score, roc_auc_score, mean_squared_error, mean_absolute_error
from sklearn import preprocessing
from copy import deepcopy
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.utils import class_weight
from torch.utils.data import RandomSampler, SequentialSampler

from collections import Counter
os.chdir("D:\\UTR\\UTR-main")

global layers, heads, embed_dim, batch_toks, cnn_layers, alphabet

layers = 6
heads = 16
embed_dim = 128
# batch_toks = 4096*2 #4096
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
repr_layers = [0, layers]

class ConvTransformerPredictor(nn.Module):
    def __init__(self, alphabet, dropout=0.2, CovTransformer_layers=3,
                 kmer=7, layers=6, embed_dim=128, nodes=40, heads=16):
        super(ConvTransformerPredictor, self).__init__()
        self.embedding_size = embed_dim
        self.nodes = nodes
        self.dropout = dropout
        self.esm2 = ESM2_SISS(num_layers = layers,
                        embed_dim = embed_dim,
                        attention_heads = heads,
                        alphabet = alphabet)
        # 修改为 nn.ModuleList
        self.convtransformer_decoder = nn.ModuleList([
            ConvTransformerLayer(embed_dim, embed_dim*4, heads, kmer-i*2, dropout=self.dropout, use_esm1b_layer_norm=True) #(kmer-i*2)
            for i in range(CovTransformer_layers)
        ])
        self.dropout = nn.Dropout(self.dropout)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        # 处理实验来源信息的线性层
        self.experiment_dense = nn.Linear(2, self.nodes)  # 处理 one-hot 实验指示符
        self.linear = nn.Linear(in_features = 6 * embed_dim, out_features = self.nodes)
        self.linear_2 = nn.Linear(in_features = self.nodes, out_features = self.nodes * 4)
        self.linear_3 = nn.Linear(in_features = self.nodes * 4, out_features = self.nodes)
        self.output = nn.Linear(in_features = self.nodes, out_features = 1)

    def forward(self, tokens, experiment_indicator, self_attn_padding_mask=None):
        # ESM embedding
        embeddings = self.esm2(tokens, repr_layers, return_representation=True)
        embeddings_rep = embeddings["representations"][layers][:, 1 : -1] #B*(T+2)*E -> B*T*E

        for i, layer in enumerate(self.convtransformer_decoder):
            x_o, attn = layer(x=embeddings_rep, self_attn_padding_mask=self_attn_padding_mask)  #tokens: B*T*E, x_o: B*T*E

        x = torch.flip(x_o, dims=[1])  # Reverse along the sequence length dimension
        # Select frames corresponding to frame 1, frame 2, and frame 3
        frame_1 = x[:, 0::3, :]
        frame_2 = x[:, 1::3, :]
        frame_3 = x[:, 2::3, :]
        # 全局最大池化
        frame_1_max = torch.max(frame_1, dim=1)[0]  # B*C
        frame_2_max = torch.max(frame_2, dim=1)[0]  # B*C
        frame_3_max = torch.max(frame_3, dim=1)[0]  # B*C
        # 扩展 self_attn_padding_mask 的维度以匹配特征张量
        mask_expanded = ~self_attn_padding_mask.unsqueeze(2)  # (batch_size, seq_len, 1)，True 表示有效数据
        # 计算有效位置的均值池化
        def masked_mean(frame, mask):
            frame_sum = torch.sum(frame * mask, dim=1)
            mask_sum = torch.sum(mask, dim=1) + 1e-8  # 避免除零
            return frame_sum / mask_sum
        # 全局均值池化
        frame_1_avg = masked_mean(frame_1, mask_expanded[:, 0::3, :])
        frame_2_avg = masked_mean(frame_2, mask_expanded[:, 1::3, :])
        frame_3_avg = masked_mean(frame_3, mask_expanded[:, 2::3, :])
        # 将池化后的张量拼接为一个张量
        pooled_output = torch.cat([frame_1_max, frame_1_avg, frame_2_max, frame_2_avg, frame_3_max, frame_3_avg], dim=1)  # B*(6*C)
        # 线性层处理实验指示符
        experiment_output = self.experiment_dense(experiment_indicator)
        x_pooled = self.flatten(pooled_output)

        o_linear = self.linear(x_pooled) + experiment_output #将池化输出与实验信息拼接
        o_linear_2 = self.linear_2(o_linear)
        o_linear_3 = self.linear_3(o_linear_2)

        o_relu = self.relu(o_linear_3)
        o_dropout = self.dropout(o_relu)
        o = self.output(o_dropout)  # B*1

        return o

    def r2(x, y):
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        return r_value ** 2

    def performances(label, pred):
        label, pred = list(label), list(pred)
        r = r2(label, pred)
        R2 = r2_score(label, pred)
        rmse = np.sqrt(mean_squared_error(label, pred))
        mae = mean_absolute_error(label, pred)
        try:
            pearson_r = pearsonr(label, pred)[0]
        except:
            pearson_r = -1e-9
        try:
            sp_cor = spearmanr(label, pred)[0]
        except:
            sp_cor = -1e-9
        print(
            f'r-squared = {r:.4f} | pearson r = {pearson_r:.4f} | spearman R = {sp_cor:.4f} | R-squared = {R2:.4f} | RMSE = {rmse:.4f} | MAE = {mae:.4f}')
        return [r, pearson_r, sp_cor, R2, rmse, mae]

    def performances_to_pd(performances_list):
        performances_pd = pd.DataFrame(performances_list, index=['r2', 'PearsonR', 'SpearmanR', 'R2', 'RMSE', 'MAE']).T
        return performances_pd

    def generate_dataset_dataloader(e_data, obj_col, lab_col, batch_toks=8192 * 4, mask_prob=0.0):
        dataset = FastaBatchedDataset(e_data.loc[:, obj_col], e_data.loc[:, lab_col], mask_prob=mask_prob)
        batches = dataset.get_batch_indices(toks_per_batch=batch_toks, extra_toks_per_seq=2)
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 collate_fn=alphabet.get_batch_converter(),
                                                 batch_sampler=batches,
                                                 shuffle=False)
        print(f"{len(dataset)} sequences")
        return dataset, dataloader, batches

    def get_experiment_indicator_for_batch(data_combine, batch_idx):
        # 从 train_combine 中获取对应 batch 的 experiment_indicator
        batch_experiment_indicators = data_combine.iloc[batch_idx]['experiment_indicator'].values.tolist()
        # 转换为 tensor
        experiment_indicator_tensor = torch.tensor(batch_experiment_indicators, dtype=torch.float32).to(device)
        return experiment_indicator_tensor

    def shuffle_data_fn(in_data):
        # 使用 sample(frac=1) 来打乱数据集顺序
        shuffle_data = in_data.sample(frac=1).reset_index(drop=True)
        return shuffle_data

    def train_step(train_dataloader, train_shuffle_combine, train_shuffle_batch, model, epoch):
        model.train()
        y_pred_list, y_true_list, loss_list = [], [], []

        for i, (labels, strs, masked_strs, toks, masked_toks, _) in enumerate(tqdm(train_dataloader)):
            toks = toks.to(device)
            padding_mask = toks.eq(alphabet.padding_idx)[:, 1:-1]
            labels = torch.FloatTensor(labels).to(device).reshape(-1, 1)
            experiment_indicator_tensor = get_experiment_indicator_for_batch(train_shuffle_combine, train_shuffle_batch[i])

            outputs = model(toks, experiment_indicator_tensor, self_attn_padding_mask=padding_mask)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_list.append(loss.cpu().detach())

            y_true_list.extend(labels.cpu().reshape(-1).tolist())
            y_pred = outputs.reshape(-1).cpu().detach().tolist()
            y_pred_list.extend(y_pred)

        loss_epoch = float(torch.Tensor(loss_list).mean())
        print(f'Train: Epoch-{epoch}/{num_epochs} | Loss = {loss_epoch:.4f} | ', end='')

        metrics = performances(y_true_list, y_pred_list)
        return metrics, loss_epoch

    def eval_step(test_dataloader, test_combine, test_batch, model, epoch):
        model.eval()
        y_pred_list, y_true_list, loss_list = [], [], []
        strs_list = []
        with torch.no_grad():
            for i, (labels, strs, masked_strs, toks, masked_toks, _) in enumerate(tqdm(test_dataloader)):
                strs_list.extend(strs)
                toks = toks.to(device)
                padding_mask = toks.eq(alphabet.padding_idx)[:, 1:-1]
                labels = torch.FloatTensor(labels).to(device).reshape(-1, 1)
                experiment_indicator_tensor = get_experiment_indicator_for_batch(test_combine, test_batch[i])

                outputs = model(toks, experiment_indicator_tensor, self_attn_padding_mask=padding_mask)
                loss = criterion(outputs, labels)
                loss_list.append(loss.cpu().detach())

                y_pred = outputs.reshape(-1).cpu().detach().tolist()
                y_true_list.extend(labels.cpu().reshape(-1).tolist())
                y_pred_list.extend(y_pred)

            loss_epoch = float(torch.Tensor(loss_list).mean())
            print(f'Test: Epoch-{epoch}/{num_epochs} | Loss = {loss_epoch:.4f} | ', end='')
            metrics = performances(y_true_list, y_pred_list)
            e_pred = pd.DataFrame([strs_list, y_true_list, y_pred_list], index=['utr', 'y_true', 'y_pred']).T

        return metrics, loss_epoch, e_pred


alphabet = Alphabet(mask_prob = 0.0, standard_toks = 'AGCT')
print(alphabet.tok_to_idx)
assert alphabet.tok_to_idx == {'<pad>': 0, '<eos>': 1, '<unk>': 2, 'A': 3, 'G': 4, 'C': 5, 'T': 6, '<cls>': 7, '<mask>': 8, '<sep>': 9}

esm2_modelfile = 'D:\\UTR\\UTR-main\Model\\utr_lm\ESM2SISS_FS4.22_fiveSpeciesCao_6layers_16heads_128embedsize_4096batchToks_lr1e-05_supervisedweight1.0_structureweight1.0_MLMLossMin_epoch115.pkl'
model = ConvTransformerPredictor(alphabet).to(device)
state_dict = torch.load(esm2_modelfile, map_location=device)
model.esm2.load_state_dict({k.replace('module.', ''):v for k,v in state_dict.items()})

num_epochs = 300

learning_rate = 1e-4 #1e-4, 1e-05

# optimizer = optim.Adam(
#     model.parameters(),
#     lr=learning_rate,
#     betas = (0.9, 0.999),
#     eps = 1e-08
# )

optimizer = torch.optim.SGD(
    model.parameters(),
    lr = learning_rate,
    momentum=0.9,
    weight_decay = 1e-4)

# criterion = torch.nn.MSELoss() #torch.nn.HuberLoss()
criterion = torch.nn.HuberLoss()

loss_best, ep_best, r2_best = np.inf, -1, -1
loss_train_dict, loss_test_dict = dict(), dict()

metrics_train_dict = dict()
metrics_test_dict = dict()

# 假设 UTR 序列的字符集为 'A', 'G', 'C', 'T'
alphabet_set = ['A', 'G', 'C', 'T']


# 初始化种群，生成若干随机序列
def initialize_population(pop_size, seq_len):
    population = [''.join(random.choices(alphabet_set, k=seq_len)) for _ in range(pop_size)]
    return population


# 适应度函数：通过模型评估序列的表现
def fitness_function(sequence, model, device):
    # 将 sequence 转换为模型所需的格式
    input_tensor = torch.tensor([[alphabet.tok_to_idx[nuc] for nuc in sequence]], dtype=torch.long).to(device)
    experiment_indicator = torch.tensor([[0.0, 1.0]], dtype=torch.float32).to(device)

    # 构造 padding mask：去掉 cls 和 eos
    padding_mask = input_tensor.eq(alphabet.padding_idx)[:, 1:-1]

    with torch.no_grad():
        output = model(input_tensor, experiment_indicator, self_attn_padding_mask=padding_mask)

    # 假设模型输出是 MRL 的预测值
    rl_prediction = output.item()
    return rl_prediction

# 选择：根据适应度选择父代
def select_parents(population, fitness_scores, num_parents):
    parents_idx = np.argsort(fitness_scores)[-num_parents:]  # 选择得分最高的个体
    return [population[i] for i in parents_idx]


# 交叉：生成新的子代
def crossover(parents, crossover_rate=0.5):
    new_population = []
    for i in range(0, len(parents), 2):
        if i + 1 < len(parents):
            parent1, parent2 = parents[i], parents[i + 1]
            if random.random() < crossover_rate:
                split_point = random.randint(0, len(parent1))
                child1 = parent1[:split_point] + parent2[split_point:]
                child2 = parent2[:split_point] + parent1[split_point:]
                new_population.extend([child1, child2])
            else:
                new_population.extend([parent1, parent2])
    return new_population


# 变异：对子代进行随机变异
def mutate(sequence, mutation_rate=0.01):
    sequence_list = list(sequence)
    for i in range(len(sequence_list)):
        if random.random() < mutation_rate:
            sequence_list[i] = random.choice(alphabet_set)
    return ''.join(sequence_list)


# 遗传算法主函数
def genetic_algorithm(model, device, pop_size=100, seq_len=100, num_generations=50, num_parents=20, mutation_rate=0.01,
                      crossover_rate=0.5):
    population = initialize_population(pop_size, seq_len)

    for generation in range(num_generations):
        # 评估当前种群的适应度
        fitness_scores = [fitness_function(seq, model, device) for seq in population]

        # 输出当前最优解
        best_idx = np.argmax(fitness_scores)
        print(
            f"Generation {generation}: Best sequence = {population[best_idx]}, Best fitness = {fitness_scores[best_idx]}")

        # 选择父代
        parents = select_parents(population, fitness_scores, num_parents)

        # 交叉生成新的子代
        offspring = crossover(parents, crossover_rate=crossover_rate)

        # 变异
        offspring = [mutate(seq, mutation_rate=mutation_rate) for seq in offspring]

        # 更新种群，使用父代和子代一起构成新的种群
        population = parents + offspring

    # 返回最终最优的序列
    return population[best_idx]


# 设置模型和设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvTransformerPredictor(alphabet).to(device)

# 加载模型参数
state_dict = torch.load(esm2_modelfile, map_location=device)
model.esm2.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})

# 运行遗传算法，优化序列
#best_sequence = genetic_algorithm(model, device, pop_size=100, seq_len=50, num_generations=10000, num_parents=20,  mutation_rate=0.01, crossover_rate=0.5)
#print(f"Optimized UTR sequence: {best_sequence}")
