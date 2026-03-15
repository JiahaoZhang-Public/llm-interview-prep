class ModelParallelScheduler:
    def __init__(self, shard_ids):
        self.shard_ids = shard_ids

    def schedule(self, request_id: str, stages):
        plan = []
        for i, stage in enumerate(stages):
            shard = self.shard_ids[i % len(self.shard_ids)]
            plan.append({"request_id": request_id, "stage": stage, "shard": shard})
        return plan
