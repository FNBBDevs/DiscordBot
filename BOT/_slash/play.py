import discord
from discord.app_commands import Group
from discord.app_commands import CommandTree
import youtube_dl
from discord import VoiceProtocol
import _utils.ytdl as yt
import _utils.clean_cache as clean
import random
import asyncio

class play(Group):
    def __init__(self, tree, guild):
        self.queue = []
        @tree.command(description="Play a video from YouTube", name="play",guild=discord.Object(id=guild))
        async def play(interaction : discord.Interaction, song: str):
            """
            Play song from a youtube channel.
            """
            clean.clean()
            await interaction.response.defer()
            # Checks to see if a user is in a voice channel
            user_channel = interaction.user.voice
            # IF THE BOT IS NOT THE VOICE CHANNEL, ADD TO CHANNEL
            if not user_channel:
                await interaction.followup.send("BRUH YOU NEED TO BE IN A VOICE CHANNEL SO I KNOW WHERE TO GO :rofl: :rofl: :rofl:")
            else:
                # SEE IF THE BOT IS IN THE CHANNEL - WE DONT NEED TO JOIN
                if interaction.guild.voice_client:
                    channel = interaction.guild.voice_client
                    # NEED TO MAKE A QUEUE
                    if channel.is_playing():
                        embed = discord.Embed(title=f"Queuing Song Request", description=f"Queue position: #{len(self.queue) + 1} in queue", color= 0x28a745)
                        embed.set_thumbnail(url="https://www.transparentpng.com/thumb/smiley/okey-sign-smiley-emoji-free-transparent-EEkKTT.png")
                        embed.set_footer(text=f'Queued By: {interaction.user.name}', icon_url= interaction.user.guild_avatar)
                        await interaction.followup.send(embed=embed)
                        self.queue.append(song)
                    else:
                        await load_song(song, channel, interaction)

                # SEND THE BOT TO THE CHANNEL
                else:
                    channel = await user_channel.channel.connect()
                    await load_song(song, channel, interaction)

        async def load_song(song: str, channel: VoiceProtocol, interaction: discord.Interaction):
            file, thumb, title, time = await yt.YTDLSource.from_url(song)
            embed = make_embed(file, thumb, title, time)
            embed.set_footer(text=f'Queued By: {interaction.user.name}', icon_url= interaction.user.guild_avatar)
            channel.play(discord.FFmpegPCMAudio(executable="ffmpeg.exe", source=file), after=lambda x: play_next(channel, interaction))
            await interaction.followup.send(embed=embed)
                
        async def get_file(url):
            file, thumb, title, time = await yt.YTDLSource.from_url(url)
            return file, thumb, title, time
        
        def make_embed(file, thumb, title, time): 
            embed = discord.Embed (
                title=f"{title}", 
                description="",
                color= 0x17a2b8 )
            embed.add_field(name='Length:', value=f"{int(time[3:5])} minutes {int(time[6:9])} seconds")
            embed.set_thumbnail(url=thumb)
            return embed
        
        def play_next(channel, interaction):
            if len(self.queue) > 0:
                queue_url = self.queue.pop(0)
                file, thumb, title, time = asyncio.run(get_file(queue_url))
                channel.play(discord.FFmpegPCMAudio(executable="ffmpeg.exe", source=file), after = lambda x: play_next(channel, interaction))

            
