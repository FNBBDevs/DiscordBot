import discord
from time import time
from discord.app_commands import Group

from _utils.time_utils import seconds_to_hms
from _utils.views import PlayingView


# Bro this needs some serious help. A fix will need to be made to this whole command <---- ?????
class Playing(Group):
    def __init__(self, tree, guild, args=None):
        @tree.command(
            description="Show what song is currently playing",
            name="playing",
            guild=discord.Object(id=guild),
        )
        async def playing(interaction: discord.Interaction):
            """
            Shows what song is playing.
            """

            user_channel = interaction.user.voice
            await interaction.response.defer()

            # IF THE BOT IS NOT THE VOICE CHANNEL, ADD TO CHANNEL
            if not user_channel:
                await interaction.followup.send(
                    "BRUH I AINT EVEN *IN* A VC RIGHT NOW! :skull: :rofl:"
                )
            # SEE IF THE BOT IS IN THE CHANNEL - WE DON'T NEED TO JOIN
            elif vcli := interaction.guild.voice_client:
                voice_channel = interaction.guild.voice_client
                if voice_channel:
                    if voice_channel.channel:
                        if vcli.is_playing():
                        
                            playing_info = interaction.client._fnbb_globals.get("playing")

                            current_time = time()

                            elasped = int(current_time - playing_info.get("started_at"))

                            hours, minutes, seconds = seconds_to_hms(elasped)

                            streaming_for = "${hours} ${minutes} ${seconds}"

                            if hours > 0:
                                streaming_for = streaming_for.replace("${hours}", f"{hours} hours")
                                if hours == 1:
                                    streaming_for = streaming_for.replace("hours", "hour")
                            else: 
                                streaming_for = streaming_for.replace("${hours}", "")
                            if minutes > 0: 
                                streaming_for = streaming_for.replace("${minutes}", f"{minutes} minutes")
                                if minutes == 1:
                                    streaming_for = streaming_for.replace("minutes", "minute")
                            else: 
                                streaming_for = streaming_for.replace("${minutes}", "")
                            if seconds > 0: 
                                streaming_for = streaming_for.replace("${seconds}", f"{seconds} seconds")
                            else: 
                                streaming_for = streaming_for.replace("${seconds}", "")
                                if seconds == 1:
                                    streaming_for = streaming_for.replace("seconds", "second")

                            embed = discord.Embed(
                                title="Currently Playing",
                                color=0x9D00FF,
                            )

                            embed.add_field(name="Song:", value=playing_info.get("title"), inline=False)

                            embed.add_field(name="Streaming for:", value=streaming_for, inline=False)

                            embed.set_thumbnail(url=playing_info.get("thumbnail"))

                            embed.set_footer(
                                text=f"Song requested by: {playing_info.get('requested_by')}",
                                icon_url=playing_info.get("requested_by_icon"),
                            )

                            await interaction.followup.send(embed=embed, view=PlayingView())
                        else:
                            await interaction.followup.send(
                                "BRUH I AINT EVEN *PLAYING MUSIC*! :skull: :rofl:"
                            )
                    else:
                        await interaction.followup.send(
                            "BRUH I AINT EVEN *PLAYING MUSIC*! :skull: :rofl:"
                        )
