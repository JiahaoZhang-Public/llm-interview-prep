# P033 Gradient Accumulation 题解

## 核心思路

1. 多个 mini-batch 的梯度累积后再做一次 optimizer.step()
2. 每个 mini-batch 的 loss 要除以 accumulation_steps（等效于用大 batch 的平均 loss）
3. 每累积 accumulation_steps 个 batch 后 step + zero_grad
4. 返回实际执行的 optimizer step 次数

## 面试口述模板

> "Gradient accumulation 通过多步累积梯度模拟大 batch 训练。
> 当显存不够放大 batch 时，把大 batch 拆成多个小 batch，
> 每个小 batch forward + backward 但不 step，
> 梯度自动累积在 .grad 上，每 N 步做一次 step。
> 关键是每步 loss 要除以 N，否则等效 learning rate 会变大。
> effective_batch_size = micro_batch_size × accumulation_steps。"

## 常见坑

- **loss 不除 accumulation_steps**：梯度累加 N 次等于 N 倍 learning rate
- **zero_grad 时机**：在 step 之后立即 zero_grad，不是在循环开始
- **尾部不满 N 步**：最后不足 accumulation_steps 的 batch 需要特殊处理
- **和 DDP 配合**：gradient accumulation 的中间步不需要 all-reduce

## 复杂度

- 和不用 accumulation 时总计算量相同
- 显存只需要放一个 micro-batch
- 训练速度略慢（多了 N-1 次 forward+backward 的 overhead）
