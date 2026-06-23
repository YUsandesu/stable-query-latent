# training.py 训练模型 Review

## 1. 入口和实际训练脚本

`training.py` 不是模型训练实现本身，而是一个 PyQt6 训练监控/启动器。点击“训练开始”或使用 `--auto-start` 后，它实际启动：

```text
VICReg_review/train_vicreg_review_h5.py
```

并固定追加以下运行参数：

```text
--checkpoint-out VICReg_review/heads/gui_run/vicreg_review_h5_latest*.pt
--best-checkpoint-out VICReg_review/heads/gui_run/vicreg_review_h5_best*.pt
--history-tsv VICReg_review/heads/gui_run/vicreg_review_h5_history*.tsv
--manifest-json VICReg_review/heads/gui_run/vicreg_review_h5_manifest*.json
--amp
```

因此，`training.py` 训练时使用的是 `VICReg_review/model.py` 中的 `LatentArrayMLP`，不是根目录的 `latent_query_model.py` / `latent_query_model_v2.py`。

默认训练数据为：

```text
VICReg_review/h5/game_review_cleaned_3_sentences.h5
```

当前本地 H5 文件属性：

| 项目 | 值 |
|---|---:|
| games | 293 |
| reviews | 407,421 |
| sentences | 8,992,181 |
| input_dim | 1024 |
| vectors dtype | float16 |
| source_shards | 8 |

每个 sentence 已经被嵌入为 1024 维向量，训练阶段不再调用文本 embedding 模型。

## 2. 训练默认参数

`training.py` 通过 GUI 启动 H5 训练时，模型和优化相关默认值来自 `VICReg_review/train_vicreg_review_h5.py`：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| epochs | 100 | 总训练 epoch |
| batch_size | 16 | 每个 batch 的 game 数 |
| steps_per_epoch | 0 | 0 表示按数据量自动计算；当前 293 games 时为 `ceil(293/16)=19` |
| sample_fraction | 0.6 | 每个 view 随机采样 60% reviews |
| cache_mode | queue | 后台线程从 H5 预取 batch |
| prefetch_batches | 2 | queue 模式预取队列长度 |
| cache_dtype | float16 | H5 读入缓存 dtype |
| pin_cache | false | 默认不 pin CPU memory |
| backward_mode | recompute | 省显存反向传播模式 |
| game_order | random | 每个 epoch 随机 game 顺序 |
| device | auto | 有 CUDA 则用 CUDA |
| amp | true | `training.py` 启动时固定传入 `--amp` |
| seed | 42 | torch 和 numpy 随机种子 |
| learning_rate | 3e-4 | AdamW |
| weight_decay | 1e-4 | AdamW |
| grad_clip | 1.0 | 对 encoder + adversary probe 全部优化参数裁剪 |

## 3. “60% random mask” 实际机制

代码里的 60% 不是 token mask，也不是 embedding 维度 mask，而是 review-level random view sampling。

对每个 game，每次训练构造两个 view：

```text
view_a = 从该 game 的所有 reviews 中随机取 ceil(review_count * 0.6) 个 review
view_b = 独立随机取 ceil(review_count * 0.6) 个 review
```

细节：

- 采样单位是 review，不是 sentence。
- 被选中的 review 会保留其全部 sentence embeddings。
- 每个 view 内部按原 review 顺序排序后拼接 sentence vectors。
- 两个 view 独立采样，所以会有重叠；期望交集约为 36% reviews，期望并集约为 84% reviews。
- 至少采 1 个 review：`max(1, ceil(review_count * sample_fraction))`。
- 由于每个 game 的 sentence 数不同，训练时每个 view 以 batch size 1 单独过 encoder，然后把多个 game 的 latent output 拼成 `(B, 256, 18)`。
- 没有 padding，因此训练路径里 `key_padding_mask=None`。

这个设计更像 self-supervised multi-view augmentation：同一个 game 的两个随机 review 子集应编码成相似表示。

## 4. Encoder: LatentArrayMLP

默认 encoder 是：

```text
LatentArrayMLP(
  input_dim=1024,
  latent_dim=256,
  num_latents=256,
  num_heads=8,
  dropout=0.1,
  output_dim=18,
  reduce_hidden=(128, 64, 32)
)
```

输入/输出：

```text
input:  (1, S, 1024)      # 一个 game view，S 是该 view 的 sentence 数
output: (1, 256, 18)      # 256 个 latent slots，每个 18 维

训练 batch 拼接后:
z_a, z_b: (B, 256, 18)    # 默认 B=16
```

完整结构：

```text
sentence embeddings
  -> LayerNorm(1024)
  -> Linear(1024, 256)
  -> LayerNorm(256)

learnable latent array
  shape = (256, 256)
  init = Normal(0, 0.02)
  -> expand to (B, 256, 256)
  -> LayerNorm(256)

CrossAttention
  query = latent array
  key/value = projected sentence embeddings
  MultiheadAttention(embed_dim=256, num_heads=8, dropout=0.1, batch_first=True)
  每个 head 维度 = 256 / 8 = 32

  -> LayerNorm(256)
  -> shared per-latent reduction MLP:
       Linear(256, 128)
       GELU
       Dropout(0.1)
       Linear(128, 64)
       GELU
       Dropout(0.1)
       Linear(64, 32)
       GELU
       Dropout(0.1)
       Linear(32, 18)
```

重要结构特征：

- 只有一层 cross-attention。
- 没有 cross-attention residual。
- 没有 latent self-attention block。
- 没有 Transformer FFN block。
- 没有位置编码。
- `reduce` MLP 对每个 latent slot 共享权重。

默认参数量：

| 模块 | 参数量 |
|---|---:|
| LayerNorm(1024) | 2,048 |
| Linear(1024,256) | 262,400 |
| latent_array(256,256) | 65,536 |
| query/context/output LayerNorm(256) | 1,536 |
| MultiheadAttention(256,8) | 263,168 |
| reduce MLP 256->128->64->32->18 | 43,826 |
| Encoder 总计 | 638,514 |

## 5. Latent Array 和 Perceiver 的区别

这个模型借用了 Perceiver 的核心想法：用固定数量的 learnable latent queries 去 cross-attend 任意长度输入，从而把长输入压缩成固定大小 latent set。

相同点：

- 都有 learnable latent array。
- 都使用 cross-attention：latent queries attend to input tokens/features。
- 输入长度可以变化，输出 latent 数固定。
- attention 复杂度从输入自注意力的 `O(S^2)` 变成 cross-attention 的 `O(S * L)`，这里 `S=sentence_count`，`L=256`。

关键区别：

| 项目 | Perceiver / Perceiver IO | 当前 LatentArrayMLP |
|---|---|---|
| Cross-attention | 通常可以多次进行 | 只做一次 |
| Latent self-attention | 核心组件，通常多层堆叠 | 完全没有 |
| Residual + FFN block | 标准 Transformer 风格 | 没有 residual，也没有 attention 后 FFN |
| Positional encoding | 常见，尤其原始输入模态需要位置 | 没有；sentence set 主要按语义聚合 |
| Decoder / output query | Perceiver IO 常有 decoder query | 没有 decoder，直接输出 latent code |
| 表达目标 | 通用输入输出架构 | 针对 game review set 的固定 latent 表示学习 |

所以这里更准确的名字是 latent-query pooling encoder 或 latent-array cross-attention encoder，而不是完整 Perceiver。

## 6. VICReg 训练目标

对同一个 game 的两个随机 60% review view，encoder 得到：

```text
z_a, z_b: (B, 256, 18)
```

VICReg loss 由三部分组成：

```text
loss = 25.0 * invariance
     + 25.0 * variance
     +  1.0 * covariance
```

### 6.1 Invariance

```text
invariance = MSE(z_a, z_b)
```

它要求同一个 game 的两个随机 review 子集得到相同或接近的 latent code。这里是逐 latent slot 对齐比较，即第 i 个 latent slot 和另一个 view 的第 i 个 latent slot 对齐。

### 6.2 Variance

先把 batch 和 latent slots 展平成样本轴：

```text
z_a_flat: (B * 256, 18)
z_b_flat: (B * 256, 18)
```

然后对每个 18 维 latent channel 计算标准差：

```text
variance_term = mean(ReLU(1 - std(channel)))
```

目的：避免所有 latent code collapse 到常数。

### 6.3 Covariance

同样使用 `(B * 256, 18)`，计算 18x18 covariance matrix，只惩罚 off-diagonal：

```text
covariance_term = sum(off_diagonal(cov)^2) / 18
```

注意：这里不是对 flatten 后的 4608 维向量做 covariance，而是把每个 latent vector 当作一个样本，只对最后的 18 个 channel 去相关。

## 7. Sentiment adversary / GRL 机制

除了 VICReg，训练还加了一个 sentiment adversarial loss。它的目标是让 game latent code 尽量不携带 SST sentiment head 能轻易读出的情绪信息。

使用的 frozen sentiment head：

```text
sst/heads/mlp4_1024_128_32_8_1_best.pt

Mlp4SentimentHead:
  Linear(1024, 128)
  GELU
  Dropout(0.2)
  Linear(128, 32)
  GELU
  Dropout(0.2)
  Linear(32, 8)
  GELU
  Dropout(0.2)
  Linear(8, 1)
  Sigmoid
```

因为 encoder 输出每个 latent 只有 18 维，而 SST head 需要 1024 维输入，所以 adversary 中有一个可训练 up-projection probe：

```text
latent code: (B, 256, 18)
  -> flatten latent vectors as (B * 256, 18)
  -> Gradient Reversal Layer
  -> Linear(18, 256, bias=False)
  -> GELU
  -> Linear(256, 1024, bias=False)
  -> L2 normalize
  -> frozen SST MLP4 sentiment head
  -> Bernoulli entropy
```

loss 中加入：

```text
total_loss = vicreg_loss + 1.0 * adversary_entropy_loss
```

GRL 的作用：

- 对 probe 来说，它正常最小化 entropy，尝试让 frozen SST head 输出更确定。
- 对 encoder 来说，GRL 反转梯度，使 encoder 最大化 entropy，尝试让 sentiment head 输出接近不确定。

GRL schedule：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| grl_lambda | 1.0 | 最终 GRL 强度 |
| grl_warmup_epochs | 5.0 | 前 5 个 epoch GRL=0，只学 VICReg / 让 probe 预热 |
| grl_ramp_epochs | 10.0 | 接下来 10 个 epoch 线性升到 1.0 |
| full strength | epoch 15 | 5 + 10 |

参数量：

| 模块 | 参数量 | 是否优化 |
|---|---:|---|
| Encoder | 638,514 | 是 |
| Adversary up-projection probe | 266,752 | 是 |
| Frozen SST MLP4 head | 135,601 | 否 |
| 默认总优化参数 | 905,266 | 是 |

## 8. 省显存 backward_mode=recompute

默认 `backward_mode=recompute` 的流程：

1. 对每个 view 先 `torch.no_grad()` forward，得到 latent output，并记录 RNG state。
2. 把所有 latent 拼成 `(B,256,18)`，在 latent 上计算 VICReg + adversary loss。
3. 对 latent tensor 反传，得到每个 game/view 的 latent gradient。
4. 恢复对应 RNG state，重新 forward 原 view。
5. 用保存的 latent gradient 对 encoder 逐 view backward。

这样不用同时保存所有长 view 的 attention 激活，显存压力明显更小。记录/恢复 RNG state 是为了让 dropout 在 replay 时和第一次 forward 一致。

## 9. Checkpoint / history / manifest

每个 epoch 后保存：

```text
model_state_dict
adversary_state_dict
optimizer_state_dict
epoch
global_step
args
metrics
model_class = "LatentArrayMLP"
num_latents
latent_dim
output_dim
input_h5
sst_checkpoint
```

`training.py` 会把 latest/best checkpoint、history TSV、manifest JSON 写入 `VICReg_review/heads/gui_run/`，并用编号避免覆盖已有文件。

当前最近一次 GUI run 的 checkpoint args 显示：

```text
epoch=39
global_step=741
batch_size=16
steps_per_epoch=19
sample_fraction=0.6
latent_dim=256
num_latents=256
output_dim=18
reduce_hidden=[128,64,32]
amp=true
grl_lambda=1.0
```

对应 manifest 在 epoch 40 记录过一次 checkpoint replace 的 `PermissionError`。这影响保存文件，不改变模型结构或训练目标。

## 10. Tag 回归头设计

Tag 回归头不是 VICReg 训练路径的一部分。它是 validation-only probe，用来诊断 frozen encoder 学到的 game representation 是否包含 Steam tags 语义。

`training.py` 默认每 5 个 encoder epochs 启动一次 probe：

```text
VICReg_review/train_tag_probe.py
```

并传入：

| 参数 | training.py 默认值 |
|---|---:|
| probe_every_epochs | 5 |
| probe_epochs | 200 |
| probe_patience | 20 |
| probe_feature_views | 8 |
| probe_log_every | 25 |
| probe_amp | false，除非显式传入 `--probe-amp` |

`train_tag_probe.py` 自身默认但 training.py 未覆盖的关键参数：

| 参数 | 默认值 |
|---|---:|
| sample_fraction | 0.6 |
| pool | flatten |
| hidden_dims | [256, 128] |
| dropout | 0.1 |
| test_ratio | 0.2 |
| learning_rate | 1e-3 |
| weight_decay | 1e-4 |
| grad_clip | 5.0 |
| max_pos_weight | 20.0 |
| pos_weight_strength | 0.25 |
| count_loss_weight | 0.1 |
| count_negative_weight | 0.02 |
| min_delta | 1e-4 |
| seed | 42 |

### 10.1 Feature extraction

对每个 game：

1. 用和训练一致的 `sample_fraction=0.6` 采样多个 random views。
2. `training.py` 默认采 `feature_views=8` 个 views。
3. Frozen encoder 编码每个 view，得到 `(256,18)`。
4. 对 8 个 view 的 code 求平均，得到该 game 的 cached feature：

```text
feature: (256, 18)
```

### 10.2 标签数据

当前 `VICReg_review/tags/tag_vocab.json`：

| 项目 | 值 |
|---|---:|
| num_tags | 219 |
| target_mode | binary |
| source | tags |
| min_count | 5 |
| num_games | 293 |
| games_without_labels | 10 |

probe 只保留至少有一个 tag 的 game。当前约为：

```text
games_with_labels = 293 - 10 = 283
train ~= 226
val ~= 57
```

### 10.3 Head 结构

默认 `TagRegressionHead`：

```text
TagRegressionHead(
  num_tags=219,
  num_latents=256,
  latent_out_dim=18,
  hidden_dims=(256, 128),
  dropout=0.1,
  pool="flatten"
)
```

flatten 后输入维度：

```text
256 * 18 = 4608
```

完整结构：

```text
encoder feature: (B, 256, 18)
  -> flatten: (B, 4608)

trunk:
  LayerNorm(4608)
  Linear(4608, 256)
  GELU
  Dropout(0.1)
  LayerNorm(256)
  Linear(256, 128)
  GELU
  Dropout(0.1)
  LayerNorm(128)

presence branch:
  Linear(128, num_tags)
  output = presence_logits

count branch:
  Linear(128, num_tags)
  output = count_logits
```

当前 219 tags 时：

```text
presence_logits: (B, 219)
count_logits:    (B, 219)
TagRegressionHead params: 1,279,286
```

presence 使用：

```text
prob = sigmoid(presence_logits)
```

count 使用：

```text
predicted_count = expm1(softplus(count_logits))
```

### 10.4 Probe loss

presence loss：

```text
BCEWithLogitsLoss(pos_weight=softened_pos_weight)
```

`pos_weight` 不是直接用完整 neg/pos ratio，而是先 clip 再软化：

```text
raw = clamp(neg / pos, max=20.0)
pos_weight = 1.0 + 0.25 * (raw - 1.0)
```

这样可以缓解多标签长尾导致的过度预测。

count loss：

```text
pred_log_counts = softplus(count_logits)
target_log_counts = log1p(raw_counts)

positive_loss = SmoothL1(pred_log_counts[present], target_log_counts[present])
negative_loss = SmoothL1(pred_log_counts[absent], 0)

count_loss = positive_loss + 0.02 * negative_loss
```

总 probe loss：

```text
loss = presence_loss + 0.1 * count_loss
```

early stopping 以 validation mAP 为主：

```text
score = val mAP
stop if no mAP improvement for patience epochs
```

同时报告：

- mAP
- micro-F1
- precision
- recall
- predicted_tags_per_game
- count_mae
- count_rmse

micro-F1 的 threshold 在 validation set 上从 `0.05, 0.10, ..., 0.95` 中选全局最佳值。

## 11. 推理路径中的 Tag head

`validation.py` 使用训练好的 encoder checkpoint 和 tag probe head：

```text
input text
  -> split sentences
  -> local Qwen embedding, 1024-d
  -> frozen VICReg encoder
  -> TagRegressionHead
  -> sigmoid presence probabilities
  -> expm1(softplus(count logits)) predicted counts
```

UI 中 tag 按 presence probability 从高到低排序。另一个 “最可能游戏” tab 会把预测 tag probability 向量和 games.json 中的 tag vector 做 cosine similarity。

## 12. 总结

`training.py` 当前训练的是一个轻量 latent-query cross-attention encoder：

```text
1024-d sentence embeddings
  -> Linear(1024,256)
  -> 256 learnable latent queries cross-attend to all selected review sentences
  -> per-latent MLP 256->128->64->32->18
  -> output (256,18)
```

训练机制是：

```text
同一 game 随机 60% reviews view A
同一 game 随机 60% reviews view B
  -> shared encoder
  -> VICReg invariance/variance/covariance
  -> sentiment GRL adversary
```

Tag 回归头是一个独立 diagnostic probe：

```text
frozen encoder feature (256,18)
  -> flatten 4608
  -> MLP 4608->256->128
  -> presence logits for 219 tags
  -> count logits for 219 tags
```

它不反向更新 VICReg encoder，只用于衡量 encoder 表示是否能线性/浅层 MLP 地解码出游戏标签语义。
