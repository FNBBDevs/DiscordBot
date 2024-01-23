import os
import discord
from enum import Enum
from _utils.bruhtts.bruhtts import say as bruhtts_say
import _utils.filters as af
from _utils.embeds import generic_colored_embed


class Members(Enum):
    Nolan = "Nolan"
    Ethan = "Ethan"
    Klim = "Klim"



class Say:
    """
    Description: adds two numbers.
    """

    def __init__(self, tree, guild, args=None):
        """
        Description: Constructor.
        """

        @tree.command(
            name="say",
            description="say something as one of the discord memebers!",
            guild=discord.Object(id=guild),
        )
        async def say(interaction: discord.Interaction, message: str, member: Members = Members.Nolan):
            """
            /say
            """
            await interaction.response.defer()
            
            print("CWD:", os.getcwd())
            
            result = await bruhtts_say(message=message, member=member, curr_dir=os.getcwd())
            
            if result:
                user_channel = interaction.user.voice
                if not user_channel:
                    await interaction.followup.send(
                        "pwease join a vc so i can twalk to you :pleading_face:"
                    )
                else:
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
                                source=f"{os.getcwd()}\\BOT\\_utils\\bruhtts\\audio\\outputs\\tts_rvc_output.wav",
                                options=None,
                            )

                            channel.play(audio_player)

                    # If the bot is not in the channel, but the user is, add the bot to the channel
                    else:
                        # Connect the bot to the voice channel
                        channel = await user_channel.channel.connect()
                        audio_player = discord.FFmpegPCMAudio(
                            executable="ffmpeg.exe",
                            source=f"{os.getcwd()}\\BOT\\_utils\\bruhtts\\audio\\outputs\\tts_rvc_output.wav",
                            options=None
                        )

                        channel.play(audio_player)
                    
                    embed = generic_colored_embed(
                        title=f"Speaking as {member.value} now!",
                        description="Playing the genereated synthesized audio.",
                        footer_text="Queued by: ",
                        footer_usr=interaction.user.global_name,
                        footer_img=interaction.user.guild_avatar,
                        color="WHITE"  
                    )
                    
                    await interaction.followup.send(embed=embed)

            else:
                embed = generic_colored_embed(
                    title=f"Attempt to speak as {member.value} failed!",
                    description="",
                    footer_text="Queued by: ",
                    footer_usr=interaction.user.global_name,
                    footer_img=interaction.user.guild_avatar,
                    color="RED"  
                )
                await interaction.followup.send(embed=embed)
