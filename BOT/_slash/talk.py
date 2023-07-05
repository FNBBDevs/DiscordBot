from enum import Enum

import _utils.embeds as embeds
import _utils.filters as af
import discord
import gtts
from discord import VoiceProtocol
from discord.app_commands import Group


class Language(Enum):
    Afrikaans = "af"
    Arabic = "ar"
    Bulgarian = "bg"
    Bengali = "bn"
    Bosnian = "bs"
    Catalan = "ca"
    Czech = "cs"
    Danish = "da"
    German = "de"
    Greek = "el"
    English = "en"
    Spanish = "es"
    Estonian = "et"
    Finnish = "fi"
    French = "fr"
    Gujarati = "gu"
    Hindi = "hi"
    Croatian = "hr"
    Hungarian = "hu"
    Indonesian = "id"
    Icelandic = "is"
    Italian = "it"
    Hebrew = "iw"
    Japanese = "ja"
    Javanese = "jw"
    Khmer = "km"
    Kannada = "kn"
    Korean = "ko"
    Latin = "la"
    Latvian = "lv"
    Malayalam = "ml"
    Marathi = "mr"
    Malay = "ms"
    Myanmar = "my"
    Nepali = "ne"
    Dutch = "nl"
    Norwegian = "no"
    Polish = "pl"
    Portuguese = "pt"
    Romanian = "ro"
    Russian = "ru"
    Sinhala = "si"
    Slovak = "sk"
    Albanian = "sq"
    Serbian = "sr"
    Sundanese = "su"
    Swedish = "sv"
    Swahili = "sw"
    Tamil = "ta"
    Telugu = "te"
    Thai = "th"
    Filipino = "tl"
    Turkish = "tr"
    Ukrainian = "uk"
    Urdu = "ur"
    Vietnamese = "vi"
    Chinese = "zh-CN"
    Taiwan = "zh-TW"
    Mandarin = "zh"


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


class Talk(Group):
    """
    Description: The music player.
    """

    def __init__(self, tree, guild):
        """
        Description: Constructor for the music player.
        """

        # Command that is called when the user types /play. Big bulky command that does a lot
        @tree.command(
            description="Type it and I'll say it",
            name="talk",
            guild=discord.Object(id=guild),
        )
        async def play(
            interaction: discord.Interaction,
            text: str,
            filter: Filters = Filters.none,
            language: Language = Language.English,
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
                    "pwease join a vc so i can twalk to you :pleading_face:"
                )

            # The user is in the voice channel, but the bot might not be
            else:
                tts = gtts.gTTS(f"{text}", lang=language)
                tts.save("tts.mp3")

                # Checks to see if the bot is in the voice channel with the user. If this is the case, it does not need to connect
                if interaction.guild.voice_client:
                    # Get the VoiceClient object to make requests to
                    channel = interaction.guild.voice_client
                    # If the channel is currently playing then we can pause the song and talk
                    if channel.is_playing():
                        # Alert the user that the song has been queued
                        pass

                    # If the bot is not playing, the song can be played without queuing
                    else:
                        audio_player = discord.FFmpegPCMAudio(
                            executable="ffmpeg.exe",
                            source="tts.mp3",
                            options=(
                                f'-vn -filter_complex "{af.audio_filters[str(filter.name)]}"'
                            ),
                        )

                        channel.play(audio_player)

                # If the bot is not in the channel, but the user is, add the bot to the channel
                else:
                    # Connect the bot to the voice channel
                    channel = await user_channel.channel.connect()
                    audio_player = discord.FFmpegPCMAudio(
                        executable="ffmpeg.exe",
                        source="tts.mp3",
                        options=(
                            f'-vn -filter_complex "{af.audio_filters[str(filter.name)]}"'
                        ),
                    )

                    channel.play(audio_player)
