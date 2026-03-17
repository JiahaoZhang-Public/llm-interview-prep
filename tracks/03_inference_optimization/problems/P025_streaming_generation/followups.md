# 面试追问

## 基础理解
- 为什么 streaming 能降低感知延迟？实际总时间变了吗？
- Generator 和 return list 的内存区别？
- EOS token 应该 yield 出去还是不 yield？

## Detokenization 挑战
- Subword token 边界如何处理？
- 何时可以安全地把 token 转成文字发给前端？
- SentencePiece 和 BPE 的 detokenization 策略差异？

## 传输协议
- SSE vs WebSocket 对比？
- FastAPI 中如何用 StreamingResponse 实现 SSE？
- 客户端断连后如何停止生成？

## 工程问题
- 如何实现 streaming 的超时机制？
- 如何在 streaming 中做内容安全过滤？
- 如何统计 token 使用量（计费）？
- 如何测试 streaming 接口？
