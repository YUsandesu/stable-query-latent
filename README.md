# Latent Query Benchmark

本项目用「latent query」结构，从句子级文本嵌入中预测每个评分维度的 1-5 分等级。
输入是按文本拆分后的句向量序列，输出是 10 个评分维度的分类结果。

## 任务定义

- 输入：每个样本是一段 pseudo text 对应的句子 embedding 序列，维度为 `input_dim=1024`
- 输出：10 个评分维度，每个维度做 5 类分类（1-5 分）
- 最终输出维度：`10 * 5 = 50`

评分列默认包括：

`psychological_meaning`, `psychological_mastery`, `psychological_curiosity`, `psychological_autonomy`, `psychological_immersion`,
`functional_progress_feedback`, `functional_ease_of_control`, `functional_audiovisual_appeal`, `functional_goals_and_rules`, `functional_challenge`

## 模型细节

核心模型定义在 [`latent_query_model.py`](./latent_query_model.py)。

### 1. 输入投影

先将每个 1024 维句向量线性投影到隐藏空间：

- `Linear(input_dim -> hidden_dim)`
- 默认 `hidden_dim=64`

### 2. 三级 latent query cross-attention

模型使用 3 个 Cross-Attention Block，latent query 数分别为：

- `32`
- `16`
- `8`

每个 block 的结构：

- 可学习 queries：`nn.Parameter(num_queries, dim)`
- `LayerNorm` 归一化 query 和 context
- `MultiheadAttention`
- 残差连接
- FFN：`LayerNorm -> Linear -> GELU -> Dropout -> Linear -> Dropout`

默认参数：

- `num_heads=8`
- `dropout=0.0`
- `mlp_ratio=4.0`

### 3. Flatten head

最后一个 block 输出的 8 个 latent 被直接 flatten：

- `LayerNorm`
- `Linear(8 * hidden_dim -> flat_dim)`
- `GELU`
- `Dropout`
- `LayerNorm`
- `Linear(flat_dim -> output_dim)`

默认 `flat_dim=128`。

### 4. 训练目标

训练脚本把每个评分维度都当成 5 类分类问题，使用：

- `CrossEntropyLoss`
- 标签范围要求是 `1..5`
- 预测时取 `argmax + 1`

## 数据管线

1. `generate_pseudo_text.py`
   - 从 `benchmark.csv` 生成伪文本
2. `embed_pseudo_text_sentences.py`
   - 用 sentence splitter + embedding model 生成句向量
3. `build_h5.py`
   - 把句向量、mask、标签打包成 HDF5
4. `test_latent_query_model.py`
   - 读取 HDF5，训练并评测模型

默认 HDF5 输入文件：

- `benchmark_sentence_latent_query_multi.h5`

## 训练设置

默认训练配置：

- `epochs=3000`
- `batch_size=128`
- `learning_rate=3e-4`
- `min_learning_rate=1e-5`
- `test_ratio=0.2`
- `seed=42`
- `split_by=score_combo`

优化器和调度器：

- `AdamW(weight_decay=1e-4)`
- `CosineAnnealingLR`

## 测试结果

当前仓库保存的最佳 checkpoint：

- `latent_query_benchmark_multi_classifier.pt`

对应配置：

- `input_dim=1024`
- `score_dim=10`
- `output_dim=50`
- `hidden_dim=64`
- `flat_dim=128`
- `query_sizes=(32, 16, 8)`
- `num_heads=8`
- `dropout=0.0`
- `split=score_combo groups=304 test_groups=61`

实验结果：

- 最佳测试准确率出现在 `epoch=307`
- `train_ce=0.0006218711`
- `test_ce=0.2100884825`
- `test_mae=0.0836363636`
- `test_accuracy=0.9436363636`

逐维测试结果：

| score_column | test_mae | test_accuracy |
|---|---:|---:|
| psychological_meaning | 0.069697 | 0.951515 |
| psychological_mastery | 0.030303 | 0.978788 |
| psychological_curiosity | 0.100000 | 0.915151 |
| psychological_autonomy | 0.066667 | 0.933333 |
| psychological_immersion | 0.112121 | 0.909091 |
| functional_progress_feedback | 0.145455 | 0.903030 |
| functional_ease_of_control | 0.045455 | 0.954545 |
| functional_audiovisual_appeal | 0.133333 | 0.912121 |
| functional_goals_and_rules | 0.063636 | 0.939394 |
| functional_challenge | 0.030303 | 0.975758 |

## 运行

训练与测试：

```bash
python test_latent_query_model.py
```

如需改参数，可查看脚本里的命令行选项，例如：

```bash
python test_latent_query_model.py --epochs 500 --batch-size 64 --device cuda
```

## 主要文件

- [`latent_query_model.py`](./latent_query_model.py)
- [`test_latent_query_model.py`](./test_latent_query_model.py)
- [`build_h5.py`](./build_h5.py)
- [`embed_pseudo_text_sentences.py`](./embed_pseudo_text_sentences.py)
- [`generate_pseudo_text.py`](./generate_pseudo_text.py)
- [`latent_query_benchmark_multi_classifier.pt`](./latent_query_benchmark_multi_classifier.pt)
- [`latent_query_training_history_multi_classifier.txt`](./latent_query_training_history_multi_classifier.txt)
- [`latent_query_per_dim_multi_classifier.txt`](./latent_query_per_dim_multi_classifier.txt)
