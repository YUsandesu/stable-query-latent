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

## 权重溯源测试

从 `pseudo_text_data_multi.csv` 里选取第 0 行文本，并直接复用
`benchmark_sentence_latent_query_multi.h5` 中对应的句子 embedding 做
`grad-times-input` 反向归因。目标类别使用模型对该维度的预测分数。

样本信息：

- `row_index=0`
- `game_id=72`
- `game_name=21`
- `genre_name=Puzzle game`
- `text_variant_index=1`
- `checkpoint=latent_query_benchmark_multi_classifier.pt`

原始文本：

> This is a Puzzle game. Another point is that the decision space is broad enough to make play feel flexible. During play, goals, systems, and guidance work together to create a very strong sense of direction. During play, the polished visual and sound design presentation strongly improves immersion and enjoyment. At the same time, the game is highly immersive and can make players lose track of time. From the player's perspective, the game provides very specific progress and reward feedback. In addition, players are easily drawn in by new constraints, surprises, and changes. From the player's perspective, the game makes player choices and actions feel meaningful. Input feedback is accurate, making the game feel very comfortable to control. In practice, the game is moderately challenging while remaining balanced. During play, the game clearly lets players feel that they are becoming better over time.

本次抽查 4 个维度，预测分数均与 CSV 标签一致：

| score_column | true_score | predicted_score | predicted_probability |
|---|---:|---:|---:|
| psychological_meaning | 4 | 4 | 0.998968 |
| psychological_autonomy | 4 | 4 | 0.998343 |
| functional_goals_and_rules | 5 | 5 | 0.999368 |
| functional_challenge | 3 | 3 | 0.997105 |

每个维度 top-10 句子归因如下。`importance` 是
`abs(grad * input)` 的句级总和，`signed` 是有符号的 `grad * input`
句级总和；正值表示推高目标 logit，负值表示压低目标 logit。

### `psychological_meaning`

| rank | sentence # | importance | signed | sentence |
|---:|---:|---:|---:|---|
| 1 | 8 | 28.469210 | -0.018509 | From the player's perspective, the game makes player choices and actions feel meaningful. |
| 2 | 2 | 23.160913 | -0.006788 | Another point is that the decision space is broad enough to make play feel flexible. |
| 3 | 4 | 20.712833 | 0.006157 | During play, the polished visual and sound design presentation strongly improves immersion and enjoyment. |
| 4 | 5 | 20.285248 | -0.009097 | At the same time, the game is highly immersive and can make players lose track of time. |
| 5 | 11 | 18.705542 | 0.001047 | During play, the game clearly lets players feel that they are becoming better over time. |
| 6 | 7 | 14.920107 | -0.007795 | In addition, players are easily drawn in by new constraints, surprises, and changes. |
| 7 | 6 | 12.899150 | -0.004846 | From the player's perspective, the game provides very specific progress and reward feedback. |
| 8 | 9 | 11.894173 | 0.003073 | Input feedback is accurate, making the game feel very comfortable to control. |
| 9 | 10 | 11.708222 | 0.000188 | In practice, the game is moderately challenging while remaining balanced. |
| 10 | 1 | 10.869076 | 0.002041 | This is a Puzzle game. |

### `psychological_autonomy`

| rank | sentence # | importance | signed | sentence |
|---:|---:|---:|---:|---|
| 1 | 2 | 26.957672 | 0.012270 | Another point is that the decision space is broad enough to make play feel flexible. |
| 2 | 8 | 19.949272 | 0.013524 | From the player's perspective, the game makes player choices and actions feel meaningful. |
| 3 | 9 | 18.452248 | 0.015421 | Input feedback is accurate, making the game feel very comfortable to control. |
| 4 | 5 | 17.724504 | -0.001987 | At the same time, the game is highly immersive and can make players lose track of time. |
| 5 | 11 | 16.456701 | 0.001752 | During play, the game clearly lets players feel that they are becoming better over time. |
| 6 | 4 | 16.389551 | 0.011282 | During play, the polished visual and sound design presentation strongly improves immersion and enjoyment. |
| 7 | 7 | 15.521434 | 0.002830 | In addition, players are easily drawn in by new constraints, surprises, and changes. |
| 8 | 3 | 14.815602 | 0.008010 | During play, goals, systems, and guidance work together to create a very strong sense of direction. |
| 9 | 6 | 13.883622 | -0.000842 | From the player's perspective, the game provides very specific progress and reward feedback. |
| 10 | 1 | 8.039020 | 0.002425 | This is a Puzzle game. |

### `functional_goals_and_rules`

| rank | sentence # | importance | signed | sentence |
|---:|---:|---:|---:|---|
| 1 | 3 | 49.745472 | 0.007319 | During play, goals, systems, and guidance work together to create a very strong sense of direction. |
| 2 | 6 | 21.964949 | 0.000058 | From the player's perspective, the game provides very specific progress and reward feedback. |
| 3 | 11 | 20.030167 | 0.013754 | During play, the game clearly lets players feel that they are becoming better over time. |
| 4 | 2 | 14.509171 | 0.002870 | Another point is that the decision space is broad enough to make play feel flexible. |
| 5 | 9 | 12.777904 | 0.004639 | Input feedback is accurate, making the game feel very comfortable to control. |
| 6 | 4 | 11.495865 | 0.003872 | During play, the polished visual and sound design presentation strongly improves immersion and enjoyment. |
| 7 | 7 | 11.023729 | 0.003466 | In addition, players are easily drawn in by new constraints, surprises, and changes. |
| 8 | 8 | 10.983618 | 0.009394 | From the player's perspective, the game makes player choices and actions feel meaningful. |
| 9 | 10 | 7.959258 | 0.004081 | In practice, the game is moderately challenging while remaining balanced. |
| 10 | 5 | 7.674747 | -0.001358 | At the same time, the game is highly immersive and can make players lose track of time. |

### `functional_challenge`

| rank | sentence # | importance | signed | sentence |
|---:|---:|---:|---:|---|
| 1 | 9 | 34.623436 | 0.010462 | Input feedback is accurate, making the game feel very comfortable to control. |
| 2 | 11 | 21.271782 | -0.006053 | During play, the game clearly lets players feel that they are becoming better over time. |
| 3 | 2 | 20.202581 | -0.002018 | Another point is that the decision space is broad enough to make play feel flexible. |
| 4 | 8 | 19.821337 | -0.007812 | From the player's perspective, the game makes player choices and actions feel meaningful. |
| 5 | 10 | 19.815519 | -0.003833 | In practice, the game is moderately challenging while remaining balanced. |
| 6 | 7 | 19.667583 | -0.012425 | In addition, players are easily drawn in by new constraints, surprises, and changes. |
| 7 | 3 | 17.551899 | -0.003004 | During play, goals, systems, and guidance work together to create a very strong sense of direction. |
| 8 | 1 | 16.997639 | 0.004039 | This is a Puzzle game. |
| 9 | 6 | 16.036831 | -0.003248 | From the player's perspective, the game provides very specific progress and reward feedback. |
| 10 | 4 | 14.870131 | -0.009054 | During play, the polished visual and sound design presentation strongly improves immersion and enjoyment. |

可复现时可运行：

```bash
python visualize_backprop_attribution.py --text "<sample text>" --score-column functional_goals_and_rules --device cpu
```

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
