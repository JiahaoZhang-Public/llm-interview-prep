# Interview Follow-ups

## Fundamental Understanding
- Why does streaming reduce perceived latency? Does actual total generation time change?
- What's the memory usage difference between a Python generator and returning a list?
- Should the EOS token be yielded or not? Do different frameworks handle this the same way?

## Detokenization Challenges
- How to handle subword token boundaries during streaming? (e.g., "Hello" split into "Hel" + "lo")
- When is it safe to convert tokens to text and send to the frontend?
- How do SentencePiece and BPE tokenizer detokenization strategies differ?

## Transport Protocols
- SSE (Server-Sent Events) vs WebSocket for streaming — comparison?
- How to implement SSE with FastAPI's StreamingResponse?
- When a client disconnects, how does the backend detect and stop generation?

## Engineering Questions
- How to implement a timeout mechanism for streaming?
- How to implement content safety filtering during streaming?
- How to track token usage during streaming (for billing)?
- How to test streaming endpoints? (unit testing strategies for generators)
