# P047 Token Rate Limiter — Solution Notes

## Core Approach

1. Implement the token bucket algorithm: a bucket holds up to `capacity` tokens
2. Tokens are added at a fixed `refill_rate` (tokens per second)
3. Each request consumes a specified number of tokens from the bucket
4. If the bucket has enough tokens, the request is allowed; otherwise, it is rejected or delayed
5. On each request, first refill tokens based on elapsed time since the last refill, then check availability

Algorithm: `tokens = min(capacity, tokens + elapsed * refill_rate)`; allow if `tokens >= cost`.

## Interview Oral Template

> "The token bucket algorithm controls request rate by maintaining a virtual
> bucket of tokens. Tokens refill at a constant rate up to a maximum capacity.
> Each request consumes some tokens -- if enough are available, the request
> proceeds and tokens are deducted. If not, the request is rate-limited.
> The capacity parameter controls burst tolerance -- a full bucket allows
> a burst of requests. The refill rate controls sustained throughput.
> On each request, I first calculate how many tokens have accumulated since
> the last check based on elapsed time, cap at capacity, then deduct."

## Common Pitfalls

- **Not refilling based on real time**: Must use actual elapsed time (not request count) for refilling -- otherwise the rate depends on request frequency
- **Integer overflow on elapsed time**: If no requests come for a long time, the refill calculation can produce very large values -- cap at capacity
- **Thread safety**: Multiple threads may check and deduct concurrently -- use a lock or atomic operations
- **Confusing with leaky bucket**: Token bucket allows bursts (up to capacity); leaky bucket enforces a strict constant rate

## Complexity

- Time: O(1) per request -- just arithmetic operations
- Space: O(1) -- stores token count, capacity, rate, and last refill timestamp
- Suitable for high-throughput APIs with millions of requests per second
