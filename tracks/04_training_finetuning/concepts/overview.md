# 知识卡片

## 为什么 causal LM 训练要 shift labels？

每个位置都预测下一个 token，所以输入和目标要错开一位。

## gradient clipping 的主要作用是什么？

防止梯度突然爆炸导致更新不稳定。

## LoRA 为什么参数高效？

它只学习低秩增量，而不是更新整块权重。

## 什么时候需要 gradient accumulation？

当显存不够但又需要更大的等效 batch size 时。
