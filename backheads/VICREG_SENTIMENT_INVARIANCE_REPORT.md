# VICReg 情感过滤与游戏身份保持实验

## 结论

当前结果强烈支持：VICReg 会削弱好评/差评推荐极性，因此它不适合作为单篇文本的好评率预测器。
游戏身份保持则是部分成立：Across the Obelisk 的三种情绪文本仍能排到同一游戏附近；Cyberpunk 2077 没能匹配回目标游戏。
所以更严谨的结论是：VICReg 有情感过滤倾向，但“鲁棒游戏身份编码器”还需要更稳定的 identity matching 或训练目标来验证。

## 方法

链路：

```text
文本 -> Qwen embedding -> frozen VICReg encoder -> full latent flatten -> z-score cosine nearest game
```

- VICReg feature cache: `C:\Users\admin\Documents\studable query latent\backheads\heads\recommendation_vicreg_features.npz`
- VICReg recommendation probe: `C:\Users\admin\Documents\studable query latent\backheads\heads\recommendation_vicreg_linear_probe.pt`
- top-k: 3

好评率 probe 的输出只作为“推荐语义是否仍可线性读出”的辅助诊断；如果输出饱和，不解释为真实 Steam 好评率。

## Nearest Game 结果

| 输入游戏 | 情感 | 句子数 | probe positive | 真实游戏排名 | 真实游戏相似度 | Top-3 nearest games |
|---|---|---:|---:|---:|---:|---|
| Cyberpunk 2077 | neutral | 100 | 0.0000 | 288 | -0.8203 | 1. Across the Obelisk (1385380), sim=0.8599<br>2. Thymesia (1343240), sim=0.8126<br>3. Bomb Rush Cyberfunk (1353230), sim=0.7882 |
| Cyberpunk 2077 | positive | 29 | 0.0000 | 273 | -0.6324 | 1. Exoprimal (1286320), sim=0.7784<br>2. Thymesia (1343240), sim=0.7667<br>3. Bomb Rush Cyberfunk (1353230), sim=0.7514 |
| Cyberpunk 2077 | negative | 32 | 0.0006 | 284 | -0.7336 | 1. Sonic Frontiers (1237320), sim=0.8077<br>2. Bomb Rush Cyberfunk (1353230), sim=0.8012<br>3. Hydroneer (1106840), sim=0.7840 |
| Across the Obelisk | neutral | 133 | 0.0113 | 1 | 0.8457 | 1. Across the Obelisk (1385380), sim=0.8457<br>2. Trepang2 (1164940), sim=0.8063<br>3. Thymesia (1343240), sim=0.7958 |
| Across the Obelisk | positive | 30 | 0.0000 | 8 | 0.6740 | 1. Everhood (1229380), sim=0.7850<br>2. Trepang2 (1164940), sim=0.7826<br>3. Exoprimal (1286320), sim=0.7612 |
| Across the Obelisk | negative | 27 | 0.0000 | 24 | 0.5243 | 1. Exoprimal (1286320), sim=0.7270<br>2. Everhood (1229380), sim=0.7145<br>3. Milk inside a bag of milk inside a bag of milk (1392820), sim=0.7007 |

## 同游戏不同情绪文本的 VICReg 相似度

| 游戏 | 文本对 | cosine similarity |
|---|---|---:|
| Across the Obelisk | neutral vs positive | 0.8136 |
| Across the Obelisk | neutral vs negative | 0.7115 |
| Across the Obelisk | positive vs negative | 0.9750 |
| Cyberpunk 2077 | neutral vs positive | 0.9022 |
| Cyberpunk 2077 | neutral vs negative | 0.9256 |
| Cyberpunk 2077 | positive vs negative | 0.9796 |

## 解释

- 好评率 probe 对单篇文本输出饱和，不能解释成真实好评率。
- positive / negative 的 probe positive 几乎没有差距，说明推荐极性很难从当前 VICReg 表示中线性读出。
- Across the Obelisk 的目标排名为 1 / 8 / 24，说明这个游戏上情绪变化后仍保留了一定身份信息。
- Cyberpunk 2077 的目标排名为 288 / 273 / 284，说明当前 identity matching 不能稳定恢复该游戏身份。

## 建议表述

> VICReg suppresses recommendation polarity in the tested representation. For some games, this helps different sentiment rewrites remain close to the same game identity, but robust identity preservation is not yet universal.

中文：

> VICReg 在当前表示中明显压制了好评/差评语义。对部分游戏，它能让不同情绪改写仍靠近同一游戏身份；但这种身份保持还不是普遍稳定的，需要进一步验证和改进。
