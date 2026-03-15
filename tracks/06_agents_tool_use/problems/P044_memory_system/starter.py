class ShortTermMemory:
    def __init__(self, max_items: int = 5):
        self.max_items = max_items
        self.items = []

    def append(self, item):
        self.items.append(item)
        if len(self.items) > self.max_items:
            self.items = self.items[-self.max_items:]

    def recent(self):
        return list(self.items)
