# P046 Model Parallel Scheduler — Solution Notes

## Core Approach

1. Assign model layers to GPU shards using round-robin pipeline parallelism
2. Given N layers and G GPUs, layer i is assigned to GPU (i % G)
3. Each GPU processes its assigned layers in sequence; data flows through GPUs in pipeline order
4. Return the assignment mapping: layer_index -> gpu_id
5. This is the simplest pipeline parallelism strategy -- more advanced approaches use interleaved schedules

## Interview Oral Template

> "Round-robin pipeline scheduling distributes model layers across GPUs in
> a cyclic pattern -- layer 0 goes to GPU 0, layer 1 to GPU 1, and so on,
> wrapping around. This balances the number of layers per GPU evenly.
> During forward pass, data flows through GPU 0's layers first, then GPU 1's,
> and so on. The main issue is pipeline bubbles -- GPUs sit idle waiting for
> their input. Micro-batching helps by splitting the batch into smaller
> chunks that can overlap across pipeline stages, improving utilization."

## Common Pitfalls

- **Uneven layer costs**: Round-robin assumes all layers have equal compute cost -- if some layers are heavier (e.g., the embedding layer), the assignment should be adjusted
- **Pipeline bubbles**: With a single micro-batch, all but one GPU are idle at any time -- need multiple micro-batches to fill the pipeline
- **Communication overhead**: Data must be transferred between GPUs at each stage boundary -- minimize cross-GPU transfers
- **Memory imbalance**: The first and last GPUs may need extra memory for input embeddings and output logits

## Complexity

- Time: O(N) to compute the assignment for N layers
- Pipeline throughput: approaches ideal with enough micro-batches; bubble ratio is (G-1)/(G-1+M) where M is micro-batch count
- Space: O(N) for the assignment mapping
