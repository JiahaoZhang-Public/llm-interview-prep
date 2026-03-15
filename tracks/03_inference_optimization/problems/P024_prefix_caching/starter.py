class PrefixCache:
    def __init__(self):
        raise NotImplementedError("Initialize this class in the starter.")

    def add_prefix(self, prefix_tokens, value):
        raise NotImplementedError("Implement P024.add_prefix().")

    def lookup(self, prefix_tokens):
        raise NotImplementedError("Implement P024.lookup().")
