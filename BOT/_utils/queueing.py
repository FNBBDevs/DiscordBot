class MusicQueueItem:
    def __init__(self, url, audio_filter, user, user_icon):
        self.url = url
        self.audio_filter = audio_filter
        self.user = user
        self.user_icon = user_icon

    def values(self):
        return (self.url, self.audio_filter, self.user, self.user_icon)


class MusicQueue:
    def __init__(self):
        self.queue = []

    def __len__(self):
        return len(self.queue)

    def add(self, item: MusicQueueItem):
        self.queue.append(item)

    def pop(self):
        if len(self.queue) > 0:
            return_val = self.queue[0]
            self.queue = self.queue[1:]
            return return_val

    def check_and_purge(self, user):
        previous_length = len(self.queue)
        self.queue = [item for item in self.queue if item.user != user]
        return previous_length - len(self.queue)


class StableQueueItem:
    def __init__(
        self,
        prompt: str = None,
        negative_prompt: str | None = None,
        quality: int = None,
        cfg_scale: float = None,
        steps: int = None,
        seed: int = None,
        upscale_model: str = None,
        sampler: str = None,
        channel = None,
        stable_id = None,
        user = None,
        user_avatar = None,
        images = None,
        client = None
    ):
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.quality = quality
        self.cfg_scale = cfg_scale
        self.steps = steps
        self.seed = seed
        self.upscale_model = upscale_model
        self.sampler = sampler
        self.channel = channel
        self.stable_id = stable_id
        self.user = user
        self.user_avatar = user_avatar
        self.images = images
        self.client = client

    def values(self):
        return (
            self.prompt,
            self.negative_prompt,
            self.quality,
            self.cfg_scale,
            self.steps,
            self.seed,
            self.upscale_model,
            self.sampler,
            self.channel,
            self.stable_id,
            self.user,
            self.user_avatar,
            self.images,
            self.client
        )
    
    def from_dict(self, queue_info: dict):
        self.prompt = queue_info["prompt"]
        self.negative_prompt = queue_info["negative_prompt"]
        self.quality = queue_info["quality"]
        self.cfg_scale = queue_info["cfg_scale"]
        self.steps = queue_info["steps"]
        self.seed = queue_info["seed"]
        self.upscale_model = queue_info["upscale_model"]
        self.sampler = queue_info["sampler"]
        self.channel = queue_info["channel"]
        self.stable_id = queue_info["stable_id"]
        self.user = queue_info["user"]
        self.user_avatar = queue_info["user_avatar"]
        self.images = queue_info["images"]
        return self


class StableQueue:
    def __init__(self):
        self.queue = []

    def __len__(self):
        return len(self.queue)

    def add(self, item: StableQueueItem):
        self.queue.append(item)

    def pop(self):
        if len(self.queue) > 0:
            return_val = self.queue[0]
            self.queue = self.queue[1:]
            return return_val

    def check_and_purge(self, user):
        previous_length = len(self.queue)
        self.queue = [item for item in self.queue if item.user != user]
        return previous_length - len(self.queue)
