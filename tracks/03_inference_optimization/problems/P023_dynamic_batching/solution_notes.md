# P023 Dynamic Batching — Solution Notes

## Core Approach

1. Maintain a FIFO queue of incoming inference requests
2. When ready to process, drain up to max_batch_size requests from the queue
3. Pad sequences in the batch to the same length and run inference together
4. Return results to each request's caller
5. Can also use a timeout — if the queue has any items after a wait period, process even if batch isn't full

## Interview Oral Template

> "Dynamic batching collects incoming requests into a FIFO queue and groups
> them into batches up to a maximum size for efficient GPU utilization.
> Instead of processing one request at a time, we wait briefly to accumulate
> requests, then pad them to equal length and run a single batched forward
> pass. This amortizes GPU kernel launch overhead and increases throughput.
> The tradeoff is slightly higher latency for individual requests due to
> queuing wait time, so production systems typically use a timeout to bound
> the maximum wait."

## Common Pitfalls

- **Unbounded queue**: Without a max queue size, memory can grow unboundedly under load — need backpressure
- **Not handling variable lengths**: Sequences in a dynamic batch have different lengths; must pad and mask correctly
- **Timeout tuning**: Too short defeats batching benefits; too long hurts latency — typical values are 5-50ms
- **Thread safety**: The queue is accessed by multiple request handlers concurrently — must be thread-safe (use `queue.Queue` or `asyncio.Queue`)

## Complexity

- Time: O(B · L_max · d) per batch, where B is batch size and L_max is the longest sequence
- Space: O(B · L_max) for the padded batch
- Throughput scales roughly linearly with batch size up to GPU memory limits
