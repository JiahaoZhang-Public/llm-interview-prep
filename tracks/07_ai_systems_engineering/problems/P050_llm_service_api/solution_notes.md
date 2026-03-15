# P050 LLM Service API — Solution Notes

## Core Approach

1. Build a FastAPI application with a `/generate` endpoint that accepts a prompt and returns generated text
2. Define request/response models using Pydantic: `GenerateRequest` (prompt, max_tokens, temperature) and `GenerateResponse` (text, usage)
3. The endpoint calls the LLM (or a mock) and returns the result as JSON
4. Add input validation, error handling, and optional parameters with defaults
5. Use `uvicorn` to serve the application

## Interview Oral Template

> "I build the LLM service as a FastAPI app with a POST /generate endpoint.
> The request body is a Pydantic model with the prompt as required and
> max_tokens and temperature as optional with sensible defaults. The handler
> validates the input, calls the generation function, and returns the result
> in a structured response including the generated text and token usage.
> FastAPI gives us automatic OpenAPI docs, request validation, and async
> support out of the box. For production, I'd add rate limiting, authentication,
> request logging, and streaming support via Server-Sent Events."

## Common Pitfalls

- **Blocking the event loop**: LLM inference is CPU/GPU-intensive -- must run it in a thread pool or background process to avoid blocking async FastAPI
- **No input validation**: Must validate prompt length, temperature range, max_tokens bounds -- malicious inputs can cause OOM or excessive compute
- **Missing error handling**: Wrap the generation call in try/except and return appropriate HTTP status codes (400 for bad input, 503 for model errors)
- **No timeout**: Long generation requests should have a timeout to prevent resource starvation

## Complexity

- Time: O(request_processing + LLM_inference) -- inference dominates
- Space: O(concurrent_requests * (input_length + output_length)) for in-flight requests
- FastAPI adds minimal overhead (~0.1ms) compared to the LLM inference time (~100ms-10s)
