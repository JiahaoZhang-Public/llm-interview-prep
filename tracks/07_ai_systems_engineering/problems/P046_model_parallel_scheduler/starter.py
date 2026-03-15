class ModelParallelScheduler:
    def __init__(self, shard_ids):
        raise NotImplementedError("Initialize this class in the starter.")

    def schedule(self, request_id: str, stages):
        raise NotImplementedError("Implement P046.schedule().")
