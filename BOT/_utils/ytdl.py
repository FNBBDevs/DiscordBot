import asyncio
import time
from functools import partial

import discord
import youtube_dl


# SHUT UP
# youtube_dl.utils.bug_reports_message = lambda: ''

# YTDL format options. Standard stuff. Nothing new here.
ytdl_format_options = {
    "prefer_ffmpeg": True,
    "format": "bestaudio/best",
    "restrictfilenames": True,
    "noplaylist": True,
    "nocheckcertificate": True,
    "quiet": False,
    "verbose": True,
    "no_warnings": False,
    "highWaterMark": 1 << 25,
    "default_search": "auto",
    "source_address": "0.0.0.0",
}

ffmpeg_options = {
    "options": '-af "bass=g=20"',
}

# FileDownloader object with specific load instructions
ytdl = youtube_dl.YoutubeDL(ytdl_format_options)


# Class used to make YTDL object that we use to stream audio
class YTDLSource(discord.PCMVolumeTransformer):
    def __init__(self, source, *, data, volume=1):
        super().__init__(source, volume)
        # Instance data for the YTDL object. Kind of pointless rn
        self.data = data
        self.title = data.get("title")
        self.url = ""
        self.thumbnail = data.get("thumbnail")
        self.seconds = data.get("duration")
        if self.seconds:
            self.format_time = time.strftime("%H:%M:%S", time.gmtime(self.seconds))

    # Makes life easier
    def __getitem__(self, item: str):
        return self.__getattribute__(item)

    # Turn the user input into a useable URL to stream
    @classmethod
    async def from_url(cls, url, *, loop=None, stream=True):
        """
        Convert the user inputted data into a useable URL as well as
        gather data concerning the YT video.
        """
        # Bruh. Gets the information from the youtube listing and sets data to a huge json payload
        loop = loop or asyncio.get_event_loop()
        prepare = partial(ytdl.extract_info, url=url, download=not stream)
        data = await loop.run_in_executor(None, prepare)

        # Check if this is a playlist
        if "entries" in data:
            # Grab the first item from the playlist
            data = data["entries"][0]

        # Get the title, thumbnail, and duration of the song
        title = data.get("title")
        thumbnail = data.get("thumbnail")
        seconds = data.get("duration")

        # If seconds were acquired, convert to a readable time
        if seconds:
            format_time = time.strftime("%H:%M:%S", time.gmtime(seconds))
        # Cheaty BS in case the data wasn't processed properly
        else:
            format_time = "00000000"

        # Send the data back as a dictionary that can be processed when the song is loaded
        return {
            "webpage": data["webpage_url"],
            "title": title,
            "thumbnail": thumbnail,
            "time": format_time,
            "data": ffmpeg_options,
        }

    # Convert the data from the queue into a useable URL (sometimes they expire, this is workaround)
    @classmethod
    async def regather_stream(cls, data, *, loop=None):
        """
        Prepare the song to be streamed.
        """
        # Similar to above, just using the previous data to reload state
        loop = loop or asyncio.get_event_loop()
        data = await loop.run_in_executor(
            None, lambda: ytdl.extract_info(url=data, download=False)
        )
        print("erm")

        # Return a valid URL that can be streamed by FFMPEG
        return data["url"]
