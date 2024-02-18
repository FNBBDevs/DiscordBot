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
        prompt: str,
        negative_prompt: str | None,
        quality: int,
        cfg_scale: float,
        steps: int,
        seed: int,
        upscale_model: str,
        sampler: str,
        channel,
        stable_id,
        user,
        user_avatar,
        images
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
            self.images
        )


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
