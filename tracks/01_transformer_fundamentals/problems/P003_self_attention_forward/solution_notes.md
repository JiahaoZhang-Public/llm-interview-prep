# P003 Self-Attention Forward 题解

## 核心思路

1. 输入 x 同时充当 Q、K、V 的来源（"self" 的含义）
2. 分别用三个线性层 W_Q、W_K、W_V 把 x 投影到 Q、K、V
3. 执行 scaled dot-product attention：softmax(QK^T / √d) · V
4. 和 P001 的区别：P001 接收已有的 Q/K/V，这里要自己做投影

## 面试口述模板

> "Self-attention 的 self 是指 Q、K、V 都来自同一个输入 x。
> 实现上就是三个线性投影 W_Q、W_K、W_V 把 x 映射到 Q、K、V，
> 然后做标准的 scaled dot-product attention。
> 这使每个位置都能和序列中所有其他位置交互，
> 是 Transformer 捕获长距离依赖的核心机制。"

## 常见坑

- **和 cross-attention 混淆**：self-attention 的 Q/K/V 来自同一输入；cross-attention 的 K/V 来自另一个序列
- **忘记 scale**：同 P001，必须除以 √d_k
- **投影矩阵共享**：Q/K/V 应该用独立的 Linear，不能共享参数

## 复杂度

- 时间：O(n² · d)，n 为序列长度
- 空间：O(n²) + 投影参数 3d²
