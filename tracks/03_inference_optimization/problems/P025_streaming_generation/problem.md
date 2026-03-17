# P025 Implement Streaming Generation

## Background

When using ChatGPT, Claude, or other LLM products, text appears **gradually** rather than all at once after a long wait. This is **streaming generation** — generating and outputting simultaneously.

**Technical implementation:**
- Backend: Python **generator (yield)** produces tokens one at a time
- Transport: **SSE (Server-Sent Events)** or WebSocket pushes tokens to the frontend in real-time
- Frontend: Incrementally appends text to the display area

```
User sends question
    ↓
Backend starts generating
    ↓ yield "Hello"
    ↓ yield " world"
    ↓ yield "!"
    ↓ yield [EOS] → stop
Frontend sees: Hello → Hello world → Hello world!
```

**Why streaming matters:**
- **User experience**: Perceived latency dramatically reduced — no dead waiting
- **TTFT**: Time to first token = prefill time; subsequent tokens appear at decode-step intervals
- **Cancellation**: Users can cancel early if the beginning isn't satisfactory, saving compute

## Objective

Implement a streaming generation function using Python generators to yield tokens one at a time.

## Interface

```python
def stream_generate(
    step_fn,              # Given token sequence → returns next token
    prompt_ids: list,     # Initial prompt token ids
    max_new_tokens: int,  # Maximum new tokens to generate
    eos_token_id=None     # EOS token id (optional)
) -> Generator[int, None, None]:
    """Yield newly generated token ids one at a time"""
```

## Examples

```python
def step_fn(tokens):
    return len(tokens) + 10

gen = stream_generate(step_fn, [1, 2], max_new_tokens=3)
list(gen)  # → [12, 13, 14]

# With EOS
def step_fn_eos(tokens):
    return 99 if len(tokens) >= 4 else len(tokens)

gen = stream_generate(step_fn_eos, [0, 1], max_new_tokens=10, eos_token_id=99)
list(gen)  # → [2, 3, 99] (stops early on EOS)
```

## Constraints

| Condition | Description |
|-----------|-------------|
| Return type | Python generator (using yield) |
| EOS handling | Yield EOS token then stop iteration |
| Without EOS | Generate max_new_tokens tokens then stop |
| Context maintenance | Each new token must be appended for step_fn to see |
| prompt_ids | Not part of the output (only yield newly generated tokens) |

## Practice Tips

1. Implement the basic for loop + yield
2. Pay attention to EOS yield timing (yield first, then break)
3. Think about streaming detokenization challenges (subword boundary issues)
