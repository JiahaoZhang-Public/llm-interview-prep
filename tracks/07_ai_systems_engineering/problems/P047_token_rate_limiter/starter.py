class TokenRateLimiter:
    def __init__(self, tokens_per_second: float, burst: float | None = None):
        raise NotImplementedError("Initialize this class in the starter.")

    def allow(self, token_count: int, now: float):
        raise NotImplementedError("Implement P047.allow().")
