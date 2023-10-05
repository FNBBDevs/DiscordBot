import asyncio
from enum import Enum

import _utils.embeds as embeds
import _utils.filters as af
import _utils.ytdl as yt
import discord
from discord import VoiceProtocol
from discord.app_commands import Group


class Filters(Enum):
    none = "none"
    sigma = "sigma"
    bassboost = "bassboost"
    earrape = "earrape"
    nuclear = "nuclear"
    softclip = "softclip"
    nightcore = "nightcore"
    pulsar = "pulsar"
    psyclip = "psyclip"
    reverb = "reverb"
    slowedandreverb = "slowedandreverb"
    lowpass = "lowpass"
    vaporwave = "vaporwave"
    POVUrGfBangsKlimWhileUrInTheBathroom = "POVUrGfBangsKlimWhileUrInTheBathroom"
    slowwwwww = "slowwwwww"
    wide = "wide"


# Used to stream songs from youtube like a BAWS
class Play(Group):
    """
    Description: The music player.
    """

    def __init__(self, tree, guild):
        """
        Description: Constructor for the music player.
        """
        # The queue that houses songs to be played (duh)
        self.queue = []

        # Command that is called when the user types /play. Big bulky command that does a lot
        @tree.command(
            description="Play a video from YouTube",
            name="play",
            guild=discord.Object(id=guild),
        )
        async def play(
            interaction: discord.Interaction, song: str, filter: Filters = Filters.none
        ):
            """
            Play song from a youtube channel.
            """
            # Await a response from an embed
            await interaction.response.defer()
            # Initializes to see if the user is in a channel
            user_channel = interaction.user.voice

            # If the user is not in a voice channel, prompt them to join a voice channel
            if not user_channel:
                await interaction.followup.send(
                    "BRUH YOU NEED TO BE IN A VOICE CHANNEL SO I KNOW WHERE TO GO"
                    " :rofl: :rofl: :rofl:"
                )

            # The user is in the voice channel, but the bot might not be
            else:
                # Checks to see if the bot is in the voice channel with the user. If this is the case, it does not need to connect
                if interaction.guild.voice_client:
                    # Get the VoiceClient object to make requests to
                    channel = interaction.guild.voice_client
                    # If the channel is currently playing then the song must be added to the queue
                    if channel.is_playing():
                        # Alert the user that the song has been queued
                        embed = embeds.on_success(
                            title="Queuing Song Request",
                            description=f"{song}",
                            footer_text="Queued By:",
                            footer_usr=interaction.user.name,
                            footer_img=interaction.user.guild_avatar,
                        )
                        # Add custom fields to generic success embed
                        embed.add_field(
                            name="Queue Position:",
                            value=f"#{len(self.queue) + 1} in queue",
                            inline=False,
                        )
                        embed.set_thumbnail(
                            url="https://www.transparentpng.com/thumb/smiley/okey-sign-smiley-emoji-free-transparent-EEkKTT.png"
                        )

                        # Send the response back to the interaction (i.e. reply)
                        await interaction.followup.send(embed=embed)
                        # Add the song to the queue
                        await add_song(song, filter)

                    # If the bot is not playing, the song can be played without queuing
                    else:
                        # Get the URL data to stream the song
                        file = await load_song(song, interaction=interaction)

                        source = await regather_stream(file)

                        # Stream the song to the channel and call the play_next function on completion
                        audio_player = discord.FFmpegPCMAudio(
                            executable="ffmpeg.exe",
                            before_options=(
                                "-reconnect 1 -reconnect_streamed 1"
                                " -reconnect_delay_max 5"
                            ),
                            options=(
                                f'-vn -filter_complex "{af.audio_filters[str(filter.name)]}"'
                            ),
                            source=source,
                        )

                        channel.play(
                            audio_player,
                            after=lambda x: (
                                print(f"ERROR: {x}")
                                if x
                                else play_next(channel, interaction, audio_player)
                            ),
                        )

                        # Load and display the custom embed for the "now playing" screen
                        await load(file, channel, interaction)

                # If the bot is not in the channel, but the user is, add the bot to the channel
                else:
                    # Connect the bot to the voice channel
                    channel = await user_channel.channel.connect()
                    # Load the song data from the URL passed in
                    file = await load_song(song, interaction=interaction)
                    # Reload the state of the song
                    source = await regather_stream(file)
                    # Load and display the custom embed
                    await load(file, channel, interaction)
                    # Stream the song to the channel and call the play_next function on completion
                    audio_player = discord.FFmpegPCMAudio(
                        executable="ffmpeg.exe",
                        before_options=(
                            "-reconnect 1 -reconnect_streamed 1"
                            " -reconnect_delay_max 5"
                        ),
                        options=(
                            f'-vn -filter_complex "{af.audio_filters[str(filter.name)]}"'
                        ),
                        source=source,
                    )
                    channel.play(
                        audio_player,
                        after=lambda x: (
                            print(f"ERROR: {x}")
                            if x
                            else play_next(channel, interaction, audio_player)
                        ),
                    )

        # Add the song data to the queue
        async def add_song(url, audio_filter):
            self.queue.append([url, audio_filter])

        # Load a song from the queue and return the URL data
        async def load_song(song: str, interaction: discord.Interaction):
            return await yt.YTDLSource.from_url(song)

        # Reload the state of a song URL from the queue incase the link went stale
        async def regather_stream(file_dict: dict):
            return await yt.YTDLSource.regather_stream(file_dict["webpage"])

        # Load and display a custom embed that is shown when a song is played
        async def load(
            file_dict: dict, channel: VoiceProtocol, interaction: discord.Interaction
        ):
            # Custom embed that is shown when a song is played
            embed = embeds.on_light(
                title="Now Playing",
                description=" ",
                footer_text="Requested by:",
                footer_usr=interaction.user.name,
                footer_img=interaction.user.guild_avatar,
            )

            # Get the embed data from the song data dictionary
            time = file_dict["time"]
            title = file_dict["title"]
            thumb = file_dict["thumbnail"]

            # Add some custom fields to the generic embed using the data from the youtube video
            embed.add_field(name="Song:", value=f"{title}", inline=False)
            embed.add_field(
                name="Length:",
                value=f"{int(time[:2])} hours {int(time[3:5])} minutes {int(time[6:9])} seconds",
                inline=False,
            )
            embed.set_thumbnail(url=thumb)
            embed.set_footer(
                text=f"Queued By: {interaction.user.name}",
                icon_url=interaction.user.guild_avatar,
            )

            # Send the embed out to the channel to show the current song being played
            await interaction.followup.send(embed=embed)

        # Play the next song in the queue
        def play_next(
            channel: VoiceProtocol,
            interaction: discord.Interaction,
            player: discord.FFmpegAudio,
        ):
            # If the queue is not empty, load the song from the front
            if len(self.queue) > 0:
                # Get the song from the front
                song_data = self.queue.pop(0)
                queue_url = song_data[0]
                filter = song_data[1]

                channel.stop()

                player._kill_process()

                # Get URL data dictionary by loading the song
                file_dict = asyncio.run(
                    load_song(queue_url, interaction=interaction))

                # Regather the URL data in case the link went bad
                source = asyncio.run(regather_stream(file_dict))

                # Grab the data items for the custom embed
                time = file_dict["time"]
                title = file_dict["title"]
                thumb = file_dict["thumbnail"]

                # Create custom embed for songs that are now playing from the queue
                embed = embeds.on_light(
                    title="Now Playing",
                    description=" ",
                    footer_text="Played by:",
                    footer_usr=interaction.user.name,
                    footer_img=interaction.user.guild_avatar,
                )

                # Add custom fields to the embed using data from youtube video
                embed.add_field(name="Song:", value=f"{title}", inline=False)
                embed.add_field(
                    name="Length:",
                    value=f"{int(time[:2])} hours {int(time[3:5])} minutes {int(time[6:9])} seconds",
                    inline=False,
                )

                embed.set_thumbnail(url=thumb)
                embed.set_footer(
                    text=f"Queued By: {interaction.user.name}",
                    icon_url=interaction.user.guild_avatar,
                )

                # Send the "now playing" embed to the channel
                interaction.client.loop.create_task(
                    interaction.channel.send(embed=embed)
                )

                audio_player = discord.FFmpegPCMAudio(
                    executable="ffmpeg.exe",
                    before_options=(
                        "-reconnect 1 -reconnect_streamed 1" " -reconnect_delay_max 5"
                    ),
                    options=(
                        f'-vn -filter_complex "{af.audio_filters[str(filter.name)]}"'
                    ),
                    source=source,
                )

                channel.play(
                    audio_player,
                    after=lambda x: (
                        print(f"ERROR: {x}")
                        if x
                        else play_next(channel, interaction, audio_player)
                    ),
                )
