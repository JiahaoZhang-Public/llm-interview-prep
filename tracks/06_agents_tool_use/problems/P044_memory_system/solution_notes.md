# P044 Memory System — Solution Notes

## Core Approach

1. Implement a sliding window buffer that stores the most recent N conversation turns
2. New messages are appended; when the buffer exceeds max size, the oldest messages are dropped
3. This provides short-term memory for the agent within a context window budget
4. The memory is passed as conversation history to each LLM call
5. Optionally preserve the system message even when older user/assistant messages are evicted

## Interview Oral Template

> "A sliding window memory keeps the most recent N messages as context for
> the agent. When a new message arrives and the buffer is full, I drop the
> oldest message to make room. This bounds the context length sent to the
> LLM, preventing context window overflow while keeping the most recent
> and presumably most relevant conversation history. For production systems,
> you'd combine this with a long-term memory store -- for example, summarizing
> evicted messages or storing key facts in a vector database for retrieval."

## Common Pitfalls

- **Losing system prompt**: The system message should be pinned and never evicted -- only user/assistant messages rotate out
- **Cutting mid-exchange**: Evicting a user message but keeping the assistant's reply (or vice versa) creates an incoherent conversation -- evict in pairs
- **Token count vs message count**: A fixed message count can still exceed the token limit if individual messages are long -- prefer token-based windowing
- **No long-term memory**: Sliding window alone means the agent forgets everything beyond the window -- important facts need a separate persistent store

## Complexity

- Time: O(1) for add/evict operations (append + pop from front)
- Space: O(N * avg_message_length) for the buffer
- The memory system itself is trivial -- the challenge is deciding what to keep and what to discard
