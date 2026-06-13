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

## 伪数据生成逻辑

伪文本由 `generate_pseudo_text.py` 从 `benchmark.csv` 合成。它不是逐字翻译原始分数，而是把每个样本的 10 个评分维度映射成一组可读的英文描述句。

### 1. 输入与分组

- 输入必须包含 `game_id` 和 10 个评分列
- 如果同一个 `game_id` 对应多行，脚本会先按游戏聚合
- 聚合方式是对每个评分列求平均，再把平均值映射回 1-5 分整数
- 聚合后会新增 `score_sample_count`，表示该游戏被平均了多少条原始记录
- `game_title` 不会写入输出

### 2. 分数到文本桶的映射

每个评分列先被转换成五档之一：

- `<= -1.8` -> `1` (`very_low`)
- `<= -0.6` -> `2` (`low`)
- `< 0.6` -> `3` (`medium`)
- `< 1.8` -> `4` (`high`)
- `>= 1.8` -> `5` (`very_high`)

然后每个维度从对应桶的模板池里随机抽一句。模板池针对不同维度分别写了正负向描述，比如：

- `psychological_meaning`
- `psychological_mastery`
- `psychological_curiosity`
- `psychological_autonomy`
- `psychological_immersion`
- `functional_progress_feedback`
- `functional_ease_of_control`
- `functional_audiovisual_appeal`
- `functional_goals_and_rules`
- `functional_challenge`

### 3. 句子结构

默认每条伪文本由两部分组成：

1. 开头句
2. 10 个评分维度句子

开头句从下面几种固定句式里随机选一个，并插入 `genre_name`：

- `This is a {genre}.`
- `Overall, this {genre} offers a clear player experience.`
- `As a {genre}, it leaves a fairly distinct impression.`
- `This {genre} has noticeable strengths and weaknesses across several play dimensions.`

每个维度句前还会随机加一个连接词，例如：

- `At the same time, `
- `In addition, `
- `During play, `
- `From the player's perspective, `
- `More specifically, `
- `In practice, `
- `Another point is that `
- 空字符串

如果前面加了连接词，脚本会把后面的句子首字母降成小写，让整句更自然。

### 4. 词汇扰动

为了减少模板痕迹，脚本会对部分关键词做轻量同义词替换，例如：

- `clear` / `specific` / `precise` / `detailed`
- `difficult` / `challenging` / `demanding`
- `smooth` / `fluid` / `responsive`
- `meaningful` / `valuable` / `rewarding`
- `rules` / `mechanics` / `systems`
- `visuals` / `art direction` / `graphics`

这一步是随机发生的，默认替换概率大约为 55%。

### 5. 顺序与变体

- 默认会打乱 10 个维度句子的顺序
- 可用 `--keep-order` 保持列顺序
- 默认会加入开头句
- 可用 `--no-opening` 去掉开头句
- 默认启用词汇扰动
- 可用 `--no-lexical-variation` 关闭
- `--variants-per-row` 可以为同一条聚合后的游戏记录生成多条不同伪文本

### 6. 输出字段

输出 CSV/JSONL 的核心字段包括：

- `generated_text`
- 原始标识字段，如 `game_id`、`genre_id`、`genre_name` 等
- 10 个评分列
- `score_sample_count`
- `text_variant_index`

### 7. 典型命令

生成单条伪文本：

```bash
python generate_pseudo_text.py --input benchmark.csv --output pseudo_text_data_one_per_game.csv
```

生成多变体版本：

```bash
python generate_pseudo_text.py --input benchmark.csv --output pseudo_text_data_multi.csv --variants-per-row 4
```

如果你后面还想复现完整流水线，一般是先生成伪文本，再做句子切分和嵌入，最后打包成 HDF5。

默认 HDF5 输入文件：

- `benchmark_sentence_latent_query_multi.h5`

## 训练设置

默认训练配置：

- `epochs=400`
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

- 最佳测试 MAE 出现在 `epoch=280`
- `train_ce=0.001557943557`
- `test_ce=0.19899169`
- `test_mae=0.08878787879`
- `test_accuracy=0.9424242424`

逐维测试结果：

| score_column | test_ce | test_mae | test_accuracy |
|---|---:|---:|---:|
| psychological_meaning | 0.159680 | 0.057576 | 0.966667 |
| psychological_mastery | 0.050443 | 0.042424 | 0.981818 |
| psychological_curiosity | 0.377497 | 0.166667 | 0.906061 |
| psychological_autonomy | 0.241912 | 0.093939 | 0.939394 |
| psychological_immersion | 0.345395 | 0.181818 | 0.872727 |
| functional_progress_feedback | 0.215814 | 0.090909 | 0.933333 |
| functional_ease_of_control | 0.081608 | 0.054545 | 0.975758 |
| functional_audiovisual_appeal | 0.171714 | 0.075758 | 0.954545 |
| functional_goals_and_rules | 0.125997 | 0.069697 | 0.939394 |
| functional_challenge | 0.219857 | 0.054545 | 0.954545 |

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
