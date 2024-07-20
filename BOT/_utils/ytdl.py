import asyncio
import time
from functools import partial
import discord
import yt_dlp as youtube_dl

ytdl_format_options = {
    "prefer_ffmpeg": True,
    "format": "bestaudio/best",
    "restrictfilenames": True,
    "noplaylist": True,
    "nocheckcertificate": True,
    "quiet": True,
    "verbose": False,
    "no_warnings": True,
    "highWaterMark": 1 << 25,
    "default_search": "auto",
    "source_address": "0.0.0.0",
}

ytdl = youtube_dl.YoutubeDL(ytdl_format_options)

class YTDLSource(discord.PCMVolumeTransformer):
    def __init__(self, source, *, data, volume=1):
        super().__init__(source, volume)
        self.data = data
        self.title = data.get("title")
        self.url = ""
        self.thumbnail = data.get("thumbnail")
        self.seconds = data.get("duration")
        if self.seconds:
            self.format_time = time.strftime("%H:%M:%S", time.gmtime(self.seconds))

    def __getitem__(self, item: str):
        return self.__getattribute__(item)

    @classmethod
    async def from_url(cls, url, *, loop=None, stream=True):
        loop = loop or asyncio.get_event_loop()
        prepare = partial(ytdl.extract_info, url=url, download=not stream)
        data = await loop.run_in_executor(None, prepare)

        if "entries" in data:
            data = data["entries"][0]

        title = data.get("title")
        thumbnail = data.get("thumbnail")
        seconds = data.get("duration")

        if seconds:
            format_time = time.strftime("%H:%M:%S", time.gmtime(seconds))
        else:
            format_time = "00000000"

        return {
            "webpage": data["webpage_url"],
            "title": title,
            "thumbnail": thumbnail,
            "time": format_time,
        }

    @classmethod
    async def regather_stream(cls, data, *, loop=None):
        loop = loop or asyncio.get_event_loop()
        data = await loop.run_in_executor(
            None, lambda: ytdl.extract_info(url=data, download=False)
        )

        return data["url"]
