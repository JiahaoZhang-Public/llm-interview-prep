# Interview Follow-ups

## Fundamental Understanding
- Are "softmax then top-k" and "top-k then softmax" the same? Why is the latter standard?
- Is K=1 exactly equivalent to greedy decoding? Prove it.
- Why does the "subtract max" trick prevent numerical overflow in softmax? Does it change the result?

## Comparative Analysis
- What problem does Top-K's fixed K have? Give a concrete example.
- Can Top-K and Top-P be used together? If so, which should be applied first?
- Top-K + Temperature: scale first or truncate first?

## Engineering & Optimization
- Full sort is O(V log V). How to optimize with partial sort to O(V + K log K)?
- How to efficiently implement top-k on GPU? (Hint: `torch.topk`)
- How should -inf or nan values in logits be handled?
