# Interview Follow-ups

## Fundamental Understanding
- What is the core goal of dynamic batching? How does latency vs throughput trade off?
- Why might requests in the same batch have different lengths? How does this affect GPU computation?
- When should flush be triggered? Timer-based vs count-based vs hybrid?

## Continuous Batching
- What is continuous batching / iteration-level scheduling?
- Compared to static batching, how much throughput improvement can continuous batching achieve? Why?
- In continuous batching, how to handle different requests being at different generation stages within one batch?

## Engineering Questions
- If enqueue and flush run in different threads, do you need a lock? What kind?
- How to implement a priority queue (VIP requests processed first)?
- How to handle request timeouts? What about requests waiting too long in the queue?
- How is vLLM's scheduler implemented? How does PagedAttention complement continuous batching?

## System Design
- To design an LLM serving system supporting 1000 QPS, how do you choose a batching strategy?
- How to dynamically adjust batch size based on request input_length and max_new_tokens?
- How to reduce memory waste from padding within a batch? (Hint: bucket by length)
