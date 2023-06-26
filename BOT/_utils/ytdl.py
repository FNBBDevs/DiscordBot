import discord
import os
from dotenv import load_dotenv
import youtube_dl
import asyncio
import time
import pprint

youtube_dl.utils.bug_reports_message = lambda: ''

ytdl_format_options = {
    'format': 'bestaudio/best',
    'restrictfilenames': True,
    'noplaylist': True,
    'nocheckcertificate': True,
    'ignoreerrors': False,
    'logtostderr': False,
    'quiet': True,
    'no_warnings': True,
    'default_search': 'auto',
    'source_address': '0.0.0.0'
}

ffmpeg_options = {
    'options': '-vn'
}

ytdl = youtube_dl.YoutubeDL(ytdl_format_options)

class YTDLSource(discord.PCMVolumeTransformer):
    def __init__(self, source, *, data, volume=0.5):
        super().__init__(source, volume)
        self.data = data
        self.title = data.get('title')
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
        data = await loop.run_in_executor(None, lambda: ytdl.extract_info(url, download=not stream))

        if 'entries' in data:
            # take first item from a playlist
            data = data['entries'][0]
        
        title = data.get('title')
        thumbnail = data.get("thumbnail")
        seconds = data.get("duration")

        if seconds:
            format_time = time.strftime("%H:%M:%S", time.gmtime(seconds))
        else:
            format_time = "00000000"

        file_data = {"webpage": data['webpage_url'], "title": title, "thumbnail":thumbnail, "time":format_time}

        return file_data
    
    @classmethod
    async def regather_stream(cls, data, *, loop=None):
        """Used for preparing a stream, instead of downloading.
        Since Youtube Streaming links expire."""

        loop = loop or asyncio.get_event_loop()
        data = await loop.run_in_executor(None, lambda: ytdl.extract_info(url=data, download=False))
        return data['url']
