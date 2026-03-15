# P050 LLM Service API 题解

## 核心思路

1. 用 FastAPI 构建 LLM 推理服务，暴露 `POST /generate` 端点
2. 请求体包含 prompt、max_tokens、temperature 等参数，用 Pydantic model 校验
3. 内部调用 LLM 推理（本地模型或远端 API），返回生成文本和 usage 统计
4. 加入请求队列 / 并发限制，防止 GPU OOM 或 API 限流
5. 支持健康检查端点 `GET /health` 和流式输出 `StreamingResponse`

## 面试口述模板

> "我会用 FastAPI 搭建 LLM 服务。核心是 /generate 端点：用 Pydantic BaseModel 定义请求 schema（prompt、max_tokens、temperature），做类型和范围校验。处理函数内部调用模型推理，返回 JSON 包含 generated_text 和 token usage。关键工程点包括：用 asyncio.Semaphore 限制并发推理数、加 timeout 防止请求挂起、用 middleware 记录延迟指标。流式场景用 StreamingResponse 配合 async generator 逐 token 返回。"

## 常见坑

- **并发控制**：GPU 推理不能无限并发，需用 Semaphore 或队列限制同时推理数
- **超时处理**：推理可能很慢，需设置请求超时并优雅返回 408/504
- **输入校验**：prompt 不能为空，max_tokens 需有上下界，temperature 在 [0, 2] 范围
- **错误处理**：模型加载失败、OOM 等异常需返回合适的 HTTP status code 和错误信息
- **流式中断**：客户端断开连接时需取消推理任务，避免浪费 GPU 资源

## 复杂度

- 时间：单次请求 O(max_tokens · L) 自回归生成，L 为序列长度（KV cache 下每步 O(L)）
- 空间：O(model_size + batch · L · d) 模型权重 + KV cache
