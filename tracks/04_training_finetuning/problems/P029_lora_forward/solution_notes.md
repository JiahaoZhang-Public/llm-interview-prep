# P029 LoRA Forward 题解

## 核心思路

1. 冻结原始权重矩阵 W (out × in)
2. 添加两个低秩矩阵：A (rank × in) 和 B (out × rank)，其中 rank << min(in, out)
3. 前向传播：`output = x @ W^T + x @ A^T @ B^T × (alpha/rank)`
4. B 初始化为零 → 训练开始时 LoRA 贡献为零，不破坏预训练模型

## 面试口述模板

> "LoRA 的核心思想是把权重更新约束在低秩空间里。
> 原始的全量微调相当于学一个 ΔW (out × in)，参数量是 O(d²)。
> LoRA 把 ΔW 分解为 B × A，其中 A 是 (rank × in)，B 是 (out × rank)，
> 参数量降到 O(rank × d)，rank 通常取 4-64，远小于 d。
> 前向传播就是 W·x + (B·A)·x × scaling，其中 scaling = alpha/rank。
> B 初始化为零保证训练初期不影响预训练行为。
> 推理时可以把 B×A 合并回 W，没有额外延迟。"

## 常见坑

- **B 初始化不为零**：会在训练开始就引入随机噪声，破坏预训练表示
- **scaling 公式搞错**：是 alpha/rank，不是 alpha 或 1/rank
- **忘记冻结原始权重**：LoRA 的前提是 W 不更新，否则就是全量微调加了额外参数
- **rank 选择**：太小表达力不够，太大失去低秩优势。实践中 rank=8-16 通常够用

## 复杂度

- 前向额外计算：O(B × rank × in + B × out × rank) = O(B × rank × (in + out))
- 可训练参数：rank × (in + out)，相比全量 in × out 大幅减少
- 推理可合并：LoRA 权重合回 W 后零额外开销
