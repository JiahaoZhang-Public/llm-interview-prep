class TokenRateLimiter:
    def __init__(self, tokens_per_second: float, burst: float | None = None):
        self.rate = tokens_per_second
        self.burst = burst if burst is not None else tokens_per_second
        self.tokens = self.burst
        self.last_time = 0.0

    def allow(self, token_count: int, now: float):
        elapsed = now - self.last_time
        self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
        self.last_time = now

        if self.tokens >= token_count:
            self.tokens -= token_count
            return True
        return False
