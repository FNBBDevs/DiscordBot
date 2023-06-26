import discord
from discord.app_commands import Group
from discord import VoiceProtocol
import _utils.ytdl as yt
import asyncio
import _utils.embeds as embeds

class play(Group):
    def __init__(self, tree, guild):
        self.queue = []
        @tree.command(description="Play a video from YouTube", name="play",guild=discord.Object(id=guild))
        async def play(interaction : discord.Interaction, song: str):
            """
            Play song from a youtube channel.
            """
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
                        embed=embeds.on_success(title="Queuing Song Request",
                                                description=f"{song}",
                                                footer_text="(Attempted) Skip By:",
                                                footer_usr=interaction.user.name,
                                                footer_img=interaction.user.guild_avatar )
                        
                        embed.add_field(name='Queue Position:', value=f"#{len(self.queue) + 1} in queue", inline=False)
                        embed.set_thumbnail(url="https://www.transparentpng.com/thumb/smiley/okey-sign-smiley-emoji-free-transparent-EEkKTT.png")
                        
                        await interaction.followup.send(embed=embed)
                        await add_song(song)
                    else:
                        file = await load_song(song)
                        #source = await regather_stream(file)
                        channel.play(discord.FFmpegPCMAudio(executable="ffmpeg.exe", source=file["webpage"]), after=lambda x: play_next(channel, interaction))
                        await load(file, channel, interaction)
                        

                # SEND THE BOT TO THE CHANNEL
                else:
                    channel = await user_channel.channel.connect()
                    file = await load_song(song)
                    source = await regather_stream(file)
                    await load(file, channel, interaction)
                    channel.play(discord.FFmpegPCMAudio(executable="ffmpeg.exe", source=source), after=lambda x: play_next(channel, interaction))

        # ADD SONG TO QUEUE
        async def add_song(url):
            self.queue.append(url)

        # LOAD SONG FROM QUEUE AND BEGIN TO PLAY
        async def load_song(song: str):
            return await yt.YTDLSource.from_url(song)
    
        async def regather_stream(file_dict: dict):
            return await yt.YTDLSource.regather_stream(file_dict["webpage"])
        
        async def load(file_dict: dict, channel: VoiceProtocol, interaction: discord.Interaction):

            embed=embeds.on_light(title="Now Playing",
                                        description=" ",
                                        footer_text="Queued by:",
                                        footer_usr=interaction.user.name,
                                        footer_img=interaction.user.guild_avatar )
            
            time = file_dict["time"]
            title = file_dict["title"]
            thumb = file_dict["thumbnail"]
                
            embed.add_field(name='Song:', value=f"{title}", inline=False)    
            embed.add_field(name='Length:', value=f"{int(time[3:5])} minutes {int(time[6:9])} seconds", inline=False)
            embed.set_thumbnail(url=thumb)
            embed.set_footer(text=f'Queued By: {interaction.user.name}', icon_url= interaction.user.guild_avatar)
            await interaction.followup.send(embed=embed)
        
        # PLAY NEXT IN THE QUEUE
        def play_next(channel, interaction: discord.Interaction):
            if len(self.queue) > 0:
                queue_url = self.queue.pop(0)

                file_dict = asyncio.run(load_song(queue_url))
                source = asyncio.run(regather_stream(file_dict))

                time = file_dict["time"]
                title = file_dict["title"]
                thumb = file_dict["thumbnail"]

                embed=embeds.on_light(title="Now Playing",
                                        description=" ",
                                        footer_text="Queued by:",
                                        footer_usr=interaction.user.name,
                                        footer_img=interaction.user.guild_avatar )
                
                embed.add_field(name='Song:', value=f"{title}", inline=False)    
                embed.add_field(name='Length:', value=f"{int(time[3:5])} minutes {int(time[6:9])} seconds", inline=False)
                embed.set_thumbnail(url=thumb)
                embed.set_footer(text=f'Queued By: {interaction.user.name}', icon_url= interaction.user.guild_avatar)

                interaction.client.loop.create_task(interaction.channel.send(embed=embed))

                channel.play(discord.FFmpegPCMAudio(executable="ffmpeg.exe", source=source), after=lambda x: play_next(channel, interaction))
