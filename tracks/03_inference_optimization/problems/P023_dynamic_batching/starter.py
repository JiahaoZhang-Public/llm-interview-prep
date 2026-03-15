class DynamicBatcher:
    def __init__(self, max_batch_size: int):
        self.max_batch_size = max_batch_size
        self.queue = []

    def enqueue(self, request):
        self.queue.append(request)

    def flush(self):
        batch = self.queue[:self.max_batch_size]
        self.queue = self.queue[self.max_batch_size:]
        return batch
