# Interview Follow-ups

## Fundamental Understanding
- Mathematically prove: why does T < 1 make the distribution sharper? (Hint: consider monotonicity and logit gap amplification)
- How should T = 0 be handled? Why does direct division fail?
- At T = 1, is temperature sampling identical to plain softmax sampling?

## Combining with Other Strategies
- Temperature + Top-K: scale first or top-k first? Does the order matter?
- Temperature + Top-P: when low T sharpens the distribution, how does the nucleus change?
- When both temperature and top_p are set in the ChatGPT API, what is the behavior?

## Engineering & Practice
- What temperature is recommended for code generation? For creative writing? Why?
- What is "temperature annealing"? In which scenarios is it used?
- How to apply different temperatures per request in batch inference?
