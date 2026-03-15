# P021 Beam Search 题解

## 核心思路

1. 维护 beam_size 个候选序列（beam），每个附带累计 log-probability
2. 每步对每个 beam 扩展所有可能的下一个 token，得到 beam_size × vocab 个候选
3. 按累计分数取 top beam_size 个作为新的 beam
4. 遇到 EOS 或达到 max_new_tokens 时停止，返回最高分序列

## 面试口述模板

> "Beam search 是介于 greedy 和穷举之间的折中搜索策略。
> 它每步保留 beam_size 个最优候选序列，不是只看当前最优（greedy），
> 而是保留多条路径，最终选全局分数最高的。
> 实现上每步把每个 beam 扩展出所有可能的 next token，
> 计算累计 log probability，然后只保留 top-B 个。
> beam_size=1 退化为 greedy，越大越接近穷举但计算量也越大。
> 注意 beam search 是确定性的，不像 sampling 有随机性。"

## 常见坑

- **用 probability 而非 log probability**：连乘概率会下溢，必须用对数相加
- **EOS 处理**：已经结束的序列不应该继续扩展，但要保留在候选里参与排序
- **长度惩罚**：不加 length normalization 会偏向短序列（短序列 log prob 绝对值更小）
- **beam_size 过大的陷阱**：并不总是越大越好，可能导致输出过于 generic

## 复杂度

- 时间：O(T × B × V)，T = 生成长度，B = beam_size，V = 每步候选数
- 空间：O(B × T) 存储 beam 序列
- 实际实现中 step_fn 通常只返回 top 若干候选，不是全 vocab
