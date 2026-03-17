# P023 Dynamic Batching Solution Notes

## Core Approach

1. Maintain a FIFO request queue
2. `enqueue` adds new requests to the tail
3. `flush` takes out up to `max_batch_size` requests as a batch
4. Removed requests are taken from the queue, preserving insertion order

This is the most basic batching strategy in LLM serving; real systems build many optimizations on top.

## Interview Oral Template

> "Dynamic batching is the foundation of LLM serving systems.
> The simplest version is a FIFO queue with a size cap: requests enqueue on arrival,
> the inference engine calls flush to grab a batch when ready.
> 
> This is better than static batching (waiting to fill N requests) because it doesn't make requests wait too long.
> But the more advanced approach is continuous batching —
> iteration-level scheduling where completed requests exit and new ones join at every decode step.
> vLLM and TGI both use this, achieving 2-3x throughput over static batching.
> 
> Real systems also consider: padding waste from varying request lengths (bucket by length),
> priority scheduling (VIP requests first), and queue timeout handling."

## Common Pitfalls

- **Empty queue flush**: Must return empty list, not raise an exception
- **Varying request lengths**: Different lengths in a batch require padding to the longest for GPU computation
- **Thread safety**: When enqueue and flush run in different threads, locking is needed (not tested here)
- **Flush timing**: Timer-based vs count-based vs whichever-comes-first → latency vs throughput tradeoff

## Complexity

- enqueue: O(1)
- flush: O(B), B = batch size taken out
- List slicing is simple but O(N) for element shifting; `collections.deque` is more efficient
