# P047 Token Rate Limiter 题解

## 核心思路

1. 使用 Token Bucket 算法限制 LLM API 的请求速率
2. 桶有最大容量 `capacity`，以固定速率 `rate`（tokens/秒）补充 token
3. 每次请求消耗若干 token，桶中不足时拒绝或等待
4. 实现：记录上次补充时间，每次请求时先计算经过时间补充 token，再扣减
5. `allow(n_tokens) -> bool`：检查并扣减；`wait(n_tokens)`：阻塞直到有足够 token

## 面试口述模板

> "Token Bucket 的实现很简洁。我维护 current_tokens 和 last_refill_time 两个变量。每次调用 allow(n) 时，先算 elapsed = now - last_refill_time，补充 elapsed * rate 个 token（不超过 capacity），更新 last_refill_time。然后检查 current_tokens >= n，满足就扣减并返回 true，否则返回 false。如果要支持阻塞等待，就计算需要等多久才能攒够 token：wait_time = (n - current_tokens) / rate，然后 sleep。"

## 常见坑

- **浮点精度**：token 数和时间都用浮点，累积误差可能导致多放或少放，建议用 int 计数
- **并发安全**：多线程环境需加锁保护 current_tokens 和 last_refill_time
- **突发请求**：桶满时允许突发等于 capacity 的请求，但之后需要等补充
- **单次请求超 capacity**：n > capacity 时永远不可能满足，需特判报错

## 复杂度

- 时间：allow O(1)，wait O(1) 加 sleep 时间
- 空间：O(1)，只需常数个变量
