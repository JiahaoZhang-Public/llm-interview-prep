# P025 Streaming Generation Solution Notes

## Core Approach

1. Use Python generator (`yield`) to output newly generated tokens one at a time
2. Each step: call `step_fn` to get next token, append to sequence (maintaining context), then yield
3. Stop on EOS or when max_new_tokens is reached
4. Callers can process tokens as they're generated (e.g., SSE push to frontend)

**Generator advantages:**
- Lazy evaluation — no need to wait for all tokens
- Memory efficient (no upfront allocation of the entire output sequence)
- Naturally supports early termination (caller stops iterating)

## Interview Oral Template

> "Streaming generation's core is using a Python generator for token-by-token output.
> In a loop, each step calls step_fn to get the next token,
> appends it to the context sequence (so subsequent step_fn calls see the history),
> then yields it out. Stops on EOS or max tokens.
> 
> In production, this generator is wrapped into SSE (Server-Sent Events)
> or WebSocket streams pushed to the frontend. Users see text appearing gradually.
> 
> Key technical details to be aware of:
> 1. Detokenization subword boundaries — a token might be half a character,
>    need to buffer until a complete character forms before sending.
> 2. Client disconnection should cancel generation to avoid wasting GPU.
> 3. Whether to yield the EOS token depends on the use case — APIs typically don't, internal interfaces may."

## Common Pitfalls

- **EOS check timing**: Yield after check, then break — ensures EOS token is output (if desired)
- **Token context append**: Must append token to sequence before the next step_fn call, otherwise step_fn can't see history
- **Detokenization trap**: Subword tokens may be partial characters (e.g., "Hel"+"lo"), sending them raw causes garbled text
- **Cancellation mechanism**: Real systems must detect client disconnection and break to stop generation
- **max_new_tokens=0**: Should return an empty generator without calling step_fn

## Complexity

- Time: O(T * C), T = generated token count, C = step_fn cost
- Space: O(T) for the growing token sequence (context keeps growing)
- Generator itself has near-zero overhead
