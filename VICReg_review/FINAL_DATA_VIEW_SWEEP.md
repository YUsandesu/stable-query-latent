# 最终数据量 x View Sweep 总览

这个文档用于收尾 Steam review VICReg 实验。目标是回答：

1. 训练可见游戏数从 50 增加到全量 293 时，性能是否稳定上升。
2. 随机 view 窗口比例 80%、60%、40%、20% 哪个更优。
3. GRL adversary 相比 no-GRL 是否带来提升。
4. compact output dim = 18 / 36 / 72 对结果有什么影响。

## 实验矩阵

默认矩阵在 `run_data_view_sweep.py` 里定义：

| 轴 | 默认值 |
|---|---|
| 训练游戏数 N | 50, 100, 150, 200, 250, 293 |
| view fraction | 0.8, 0.6, 0.4, 0.2 |
| output dim | 18, 36, 72 |
| arm | grl, nogrl |

总组合数是 6 x 4 x 3 x 2 = 144。

训练子集固定保留 Cyberpunk 2077 / Across the Obelisk 两个锚点 appid：
`1091500,1385380`。这样身份召回测试不会变成目标游戏未见过的外推测试。

## 训练方式

推荐直接跑总入口：

```powershell
C:/Users/admin/anaconda3/envs/cuda_Vit/python.exe VICReg_review/run_data_view_sweep.py
```

如果这是最终归档用的干净全矩阵重跑，建议显式重训并重评估：

```powershell
C:/Users/admin/anaconda3/envs/cuda_Vit/python.exe VICReg_review/run_data_view_sweep.py `
  --force-train --rebuild-eval
```

断点续跑直接重复同一命令即可。规则：

- 如果 `grl` 和 `nogrl` 对应组合都缺 checkpoint，脚本会调用 `train_vicreg_review_h5_paired.py`，用同一个 sampled batch 同时训练两条 arm。
- 如果某一条 arm 已有半路 checkpoint，脚本会回退到单 arm trainer 并传入 `--resume-checkpoint`。
- 只有 `vicreg_review_h5_manifest.json` 中 `status == done` 的组合会进入最终曲线和最佳窗口判断。
- 半路评估不会进入主曲线，但会保留在 raw detail 表里供排查。

如果只想先跑一个小范围验证：

```powershell
C:/Users/admin/anaconda3/envs/cuda_Vit/python.exe VICReg_review/run_data_view_sweep.py `
  --output-dims 18 --train-game-counts 50 --sample-fractions 0.8
```

## 产物

所有实验产物默认在 `VICReg_review/heads/data_view_sweep/`，该目录被 `.gitignore` 忽略。

| 文件 | 内容 |
|---|---|
| `DATA_VIEW_SWEEP_REPORT.md` | 最终总览报告，含最佳窗口、数据量曲线、GRL 差值表 |
| `data_view_sweep_summary.csv` | 每个组合的一行聚合指标 |
| `data_view_sweep_summary.json` | summary + sweep 参数 |
| `raw_test_data/eval_reports_full.jsonl` | 每个组合完整 `eval_report.json` 的合并 JSONL |
| `raw_test_data/training_manifests.csv` | 训练状态、训练子集 appid、最后 epoch 指标 |
| `raw_test_data/probe_summary.csv` | TAG probe 与情感 probe 聚合指标 |
| `raw_test_data/recommendation_probe.csv` | 好评率/差评率线性 probe 指标 |
| `raw_test_data/identity_summary.csv` | PR、rank、Hit@K、同游戏文本 cosine 聚合 |
| `raw_test_data/identity_retrieval_details.csv` | 每条测试文本的 raw/VICReg rank 与 similarity |
| `raw_test_data/identity_pair_cosine_details.csv` | 同游戏不同情绪文本 pairwise cosine |
| `raw_test_data/tag_freq_floor_details.csv` | TAG frequency floor 分层 |
| `raw_test_data/tag_top_bottom_details.csv` | 每组合 TAG probe top/bottom tags |
| `raw_test_data/tag_fold_details.csv` | TAG probe 每 fold micro-F1 |

## 指标解释

主报告里的综合分只用于窗口选择，不是论文指标：

```text
score =
  0.30 * TAG micro-F1
+ 0.30 * identity Hit@5
+ 0.15 * normalized same-game text cosine
+ 0.15 * min(PR / 25, 1)
- 0.05 * max(sentiment R2, 0)
- 0.05 * abs(recommendation Pearson)
```

解释：

- TAG micro-F1 越高越好，代表内容/类型信息保留。
- identity Hit@5 越高越好，代表测试文本能召回目标游戏。
- same-game text cosine 越高越好，代表同游戏不同情绪文本仍靠近。
- PR 越高越好，代表 compact centroid 没有低秩坍缩。
- sentiment R² 越低越好，代表情感信息不容易从 code 线性读出。
- abs(recommendation Pearson) 越低越好，代表好评率捷径被压低。

GRL 差值表定义为 `GRL - no-GRL`。因此：

- `delta_tag`、`delta_hit_at_5`、`delta_pr` 为正表示 GRL 更好。
- `delta_sentiment_r2` 为负表示 GRL 更好。
- `delta_reco_pearson_abs` 为负表示 GRL 更好。

## 最终报告判读顺序

1. 先看 `DATA_VIEW_SWEEP_REPORT.md` 的“结论”和“View 最佳窗口预测”。
2. 再看“数据量-性能曲线”，判断 N 从 50 到 293 是否整体上升。
3. 然后看“GRL 对照差值”，确认 GRL 的收益是否稳定。
4. 如果某个点异常，再去 `raw_test_data/` 查逐文本召回、TAG floor、fold 波动和训练 manifest。
