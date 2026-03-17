# Interview Follow-ups

## Fundamental Understanding
- Why use log probability instead of probability? What happens with regular probability?
- Is beam_size=1 exactly equivalent to greedy decoding? Argue why.
- Does beam search guarantee finding the globally optimal sequence? Why not?

## Advanced Variants
- What is length normalization? Why does beam search without length penalty favor shorter sequences? What formula is commonly used?
- What is diverse beam search? How to prevent multiple beams from converging to similar sequences?
- How to implement constrained beam search (e.g., must include a specific token)?

## Comparison & Selection
- Why is beam search common in translation but not in dialogue generation?
- Does a hybrid of beam search + sampling exist? (Hint: stochastic beam search)
- In production deployment, how much does beam search affect latency? Why doesn't ChatGPT use beam search?

## Engineering Implementation
- How to parallelize multiple beams using batch inference? (Hint: pack all beams into one batch)
- Memory consumption: does each beam store the full sequence? How to optimize?
- When step_fn returns the full vocab, the expansion stage has B*V candidates. How to efficiently select top-B?
