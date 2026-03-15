class DynamicBatcher:
    def __init__(self, max_batch_size: int):
        raise NotImplementedError("Initialize this class in the starter.")

    def enqueue(self, request):
        raise NotImplementedError("Implement P023.enqueue().")

    def flush(self):
        raise NotImplementedError("Implement P023.flush().")
