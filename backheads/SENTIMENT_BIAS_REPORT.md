# 真实文本情感色彩输入对 VICReg 好评率线性探针的偏差测试

## 目的

测试同一款游戏在输入不同感情色彩的单篇“真实文本”后，经过 frozen VICReg encoder，再由好评率线性探针读取时会产生多大偏差。

当前测试链路是：

```text
文本 -> Qwen embedding -> frozen VICReg encoder -> pool(stats) -> ridge linear probe -> positive_rate
negative_rate = 1 - positive_rate
```

这份报告关注的是 **经过 VICReg 后，好评率/情感相关信息还能线性恢复到什么程度**。

## 线性恢复能力

同一批 Steam 好评率标签下，对比两种 frozen feature：

| 特征来源 | feature dim | 5-fold CV MAE | final holdout MAE | final Pearson |
|---|---:|---:|---:|---:|
| raw Qwen mean+std 聚合 | 2048 | 0.0434 ± 0.0039 | 0.0380 | 0.9236 |
| VICReg code stats pool | 36 | 0.0666 ± 0.0034 | 0.0764 | 0.6961 |

结论：经过 VICReg 后，好评率信号仍可线性恢复一部分，但明显弱于 raw Qwen 聚合特征。

## 数据与方法

测试文件：

| 游戏 | 情感标签 | 文件 | 分句数 | 字符数 |
|---|---|---:|---:|---:|
| Cyberpunk 2077 | neutral | `2077_text.txt` | 100 | 3469 |
| Cyberpunk 2077 | positive | `2077_text_postive.txt` | 29 | 1470 |
| Cyberpunk 2077 | negative | `2077_text_negative.txt` | 32 | 1619 |
| Across the Obelisk | neutral | `AO_text.txt` | 133 | 3578 |
| Across the Obelisk | positive | `AO_text_postive.txt` | 30 | 1665 |
| Across the Obelisk | negative | `AO_text_negative.txt` | 27 | 1451 |

注：文件名里使用的是现有拼写 `postive`。

运行方式：

```powershell
& 'C:/Users/admin/anaconda3/envs/cuda_Vit/python.exe' backheads/predict_text_vicreg_recommendation.py `
  2077_text.txt 2077_text_postive.txt 2077_text_negative.txt `
  AO_text.txt AO_text_postive.txt AO_text_negative.txt `
  --device cuda --max-sentences 4096
```

评价文本基准来自：

```text
backheads/heads/recommendation_vicreg_features.npz
backheads/heads/recommendation_vicreg_linear_probe.pt
```

## 总体结果

| 游戏 | 输入情感 | Text positive | Text negative | Review positive | True positive | Text - Review | Text - True |
|---|---|---:|---:|---:|---:|---:|---:|
| Cyberpunk 2077 | neutral | 0.0000 | 1.0000 | 0.7162 | 0.7440 | -0.7162 | -0.7440 |
| Cyberpunk 2077 | positive | 0.0000 | 1.0000 | 0.7162 | 0.7440 | -0.7162 | -0.7440 |
| Cyberpunk 2077 | negative | 0.0006 | 0.9994 | 0.7162 | 0.7440 | -0.7156 | -0.7434 |
| Across the Obelisk | neutral | 0.0113 | 0.9887 | 0.9262 | 0.8195 | -0.9149 | -0.8082 |
| Across the Obelisk | positive | 0.0000 | 1.0000 | 0.9262 | 0.8195 | -0.9262 | -0.8195 |
| Across the Obelisk | negative | 0.0000 | 1.0000 | 0.9262 | 0.8195 | -0.9262 | -0.8195 |

## 按游戏分析

### Cyberpunk 2077

基准：

- Review positive：0.7162
- True positive：0.7440
- Steam 标签统计：19561 好评，6729 差评

情感输入影响：

| 对比 | 好评率变化 |
|---|---:|
| positive - negative | -0.0006 |
| neutral - negative | -0.0006 |
| positive - neutral | 0.0000 |

观察：

- 评价文本聚合经 VICReg 后仍能恢复到接近真实值：0.7162 vs 0.7440。
- 单篇 neutral / positive / negative 文本经过 VICReg 后几乎都被线性探针判成低好评率。
- positive 和 negative 的差距几乎消失，说明显式情感极性在这个 VICReg 表示和 stats-pool 线性读出中基本不可分。

### Across the Obelisk

基准：

- Review positive：0.9262
- True positive：0.8195
- Steam 标签统计：9095 好评，2003 差评

情感输入影响：

| 对比 | 好评率变化 |
|---|---:|
| positive - negative | 0.0000 |
| neutral - negative | +0.0113 |
| positive - neutral | -0.0113 |

观察：

- 评价文本聚合经 VICReg 后预测偏高：0.9262 vs 0.8195。
- 单篇 positive / negative 文本都被压到接近 0。
- neutral 稍高，但也只有 0.0113。
- 这同样说明单篇文本的情感差异没有被 probe 有效读取。

## 关键结论

1. **经过 VICReg 后，好评率线性可恢复能力明显下降。**
   - raw Qwen 聚合 CV MAE：0.0434
   - VICReg code CV MAE：0.0666

2. **评价文本聚合仍保留一部分群体好评率信号。**
   - Cyberpunk 的 VICReg review prediction 仍接近真实值。
   - Across 的 VICReg review prediction 有偏高，但仍在同一数量级。

3. **单篇真实文本进入 VICReg 后明显 OOD。**
   - 单篇文本不是训练 probe 时的输入分布。
   - 训练时的 feature 来自每个游戏大量 review/简介句子采样后的 VICReg 聚合表示。
   - 单篇整理文本经 VICReg 后落到线性探针不熟悉的区域，因此输出几乎饱和到低好评率。

4. **显式情感信息被严重削弱。**
   - Cyberpunk positive vs negative 差距约 0.0006。
   - Across positive vs negative 差距约 0。
   - 这比 raw Qwen 聚合读出时的 9 到 10 个百分点差距小得多。

## 对 validation 的含义

validation 现在默认使用：

```text
backheads/heads/recommendation_vicreg_linear_probe.pt
backheads/heads/recommendation_vicreg_features.npz
```

因此它显示的是：

- 当前输入文本经过 VICReg 后的 probe 预测值。
- 3 个最相似游戏在 VICReg 评价文本聚合特征上的 probe 预测值。
- 这些相似游戏的 CSV 真实好评/差评率。

但需要注意：**当前输入文本的绝对好评率不应直接解释为 Steam 好评率**，因为单篇文本经 VICReg 后存在明显分布外问题。它更适合用于观察“VICReg 表示里还剩多少好评率/情感可读信息”。

## 建议

- 如果目标是预测真实 Steam 好评率，优先使用大量评价文本聚合，而不是单篇文本。
- 如果目标是测 VICReg 是否去除了情感因素，当前测试支持“情感信息被显著削弱”的判断。
- 后续可以加一个专门的 sentiment probe：
  - 训练目标：positive / negative 文本标签。
  - 输入：VICReg code。
  - 指标：二分类准确率 / AUC。
  - 这样能直接量化 VICReg 表示里残留多少情感极性，而不是间接看好评率。
