# P023 Implement Dynamic Batching

## Background

In LLM serving, requests arrive **asynchronously**. Processing one at a time wastes GPU utilization;
waiting to fill a fixed batch size adds too much latency.

**Dynamic batching** is the compromise: maintain a request queue and flexibly form batches based on current queue state.

```
Timeline:
  t=0   Request A arrives → enqueue
  t=1   Request B arrives → enqueue
  t=2   flush → take [A, B] as batch, send to GPU
  t=3   Request C arrives → enqueue
  t=4   Requests D,E arrive → enqueue
  t=5   flush → take [C, D, E] as batch
```

**More advanced: Continuous Batching**
- Don't wait for the entire batch to finish — requests that generate EOS exit immediately
- New requests can join the running batch at the next iteration
- This is the core scheduling strategy in vLLM, TGI, and other inference frameworks

This problem implements the most basic dynamic batching: FIFO queue with max-size flush.

## Objective

Implement a DynamicBatcher class supporting enqueue and flush.

## Interface

```python
class DynamicBatcher:
    def __init__(self, max_batch_size: int):
        """Initialize with maximum batch size per flush"""

    def enqueue(self, request):
        """Add a request to the end of the queue"""

    def flush(self) -> list:
        """Return up to max_batch_size requests in FIFO order.
        Returns empty list if queue is empty."""
```

## Examples

```python
batcher = DynamicBatcher(max_batch_size=3)
batcher.enqueue({"id": "a"})
batcher.enqueue({"id": "b"})
batch = batcher.flush()  # [{"id": "a"}, {"id": "b"}]

batcher.enqueue({"id": "c"})
batcher.enqueue({"id": "d"})
batcher.enqueue({"id": "e"})
batcher.enqueue({"id": "f"})
batch = batcher.flush()  # [{"id": "c"}, {"id": "d"}, {"id": "e"}] (max 3)
batch = batcher.flush()  # [{"id": "f"}] (remainder)
```

## Constraints

| Condition | Description |
|-----------|-------------|
| max_batch_size | >= 1 |
| Empty queue flush | Returns `[]` |
| FIFO order | Earlier enqueued requests are flushed first |
| Partial flush | Can return fewer than max_batch_size |
| Request type | Any Python object |

## Practice Tips

1. Implement with list or collections.deque
2. Think: if enqueue and flush run in different threads, what synchronization is needed?
3. Verbally explain dynamic batching vs continuous batching
