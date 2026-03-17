# P025 实现 Streaming Generation

## 背景

用户使用 ChatGPT/Claude 时文字**逐渐出现**，这就是 streaming generation。

**技术栈：** Python generator（yield） → SSE/WebSocket → 前端增量显示

为什么重要：感知延迟大幅降低，支持提前取消。

## 接口规范

```python
def stream_generate(step_fn, prompt_ids: list, max_new_tokens: int, eos_token_id=None):
    """Generator：逐个 yield 新生成的 token id"""
```

step_fn(tokens) → next_token_id

## 约束

| 条件 | 说明 |
|------|------|
| 返回类型 | Generator（yield） |
| EOS 处理 | yield EOS 后停止 |
| 上下文维护 | 每个新 token 要 append 到序列 |
| prompt_ids | 不作为输出 |

## 练习建议

1. for 循环 + yield
2. 注意 EOS yield 时机
3. 思考 streaming detokenization 的挑战
