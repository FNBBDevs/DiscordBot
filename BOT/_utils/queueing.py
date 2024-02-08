

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
