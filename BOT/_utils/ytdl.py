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

    @classmethod
    async def from_url(cls, url, *, loop=None, stream=False):
        loop = loop or asyncio.get_event_loop()
        data = await loop.run_in_executor(None, lambda: ytdl.extract_info(url, download=not stream))

        if 'entries' in data:
            # take first item from a playlist
            data = data['entries'][0]
            thumbnail = data["thumbnail"]
            title = data["title"]
            seconds = data["duration"]
            format_time = time.strftime("%H:%M:%S", time.gmtime(seconds))
        else:
            thumbnail="https://pbs.twimg.com/media/EAmr-PAWsAEoiWR.jpg"
            format_time="00000000"
            title="UNKNOWN"

        filename = data['title'] if stream else ytdl.prepare_filename(data)
        return filename, thumbnail, title, format_time