# Interview Follow-ups

## Fundamental Understanding
- Why divide positive logits but multiply negative ones? What goes wrong with a uniform operation?
- What happens when repetition penalty = 1.0? What about < 1.0?
- What happens to a token with logit = 0 after penalty? Is this reasonable?

## Concept Differentiation
- Repetition penalty vs Frequency penalty vs Presence penalty — what are the differences?
  - Repetition penalty: binary (appeared or not), ignores count
  - Frequency penalty: linear penalty by occurrence count (OpenAI API)
  - Presence penalty: fixed penalty if appeared at all (OpenAI API)
- Can these three penalties be combined? What are the caveats?

## Engineering Implementation
- What should the execution order of multiple logits processors (repetition penalty, temperature, top-k) be? Why?
- How to write a custom processor in HuggingFace's `LogitsProcessorList`?
- Should the EOS token be affected by repetition penalty? What happens if it's over-penalized?
- How to apply different penalties per request in batch inference?
