import discord
from discord.app_commands import Group
from _utils.embeds import generic_colored_embed


class Pause(Group):
    def __init__(self, tree, guild, args=None):
        @tree.command(
            description="Pause  the currently playing song.", name="pause", guild=discord.Object(id=guild)
        )
        async def pause(interaction: discord.Interaction):
            """
            Pause the currently playing song
            """

            user_channel = interaction.user.voice

            voice_channel = interaction.guild.voice_client

            await interaction.response.defer()

            if user_channel:
                if voice_channel:
                    if voice_channel.is_playing():

                        voice_channel.pause()

                        embed = generic_colored_embed(
                            title="Song has been paused!",
                            description="",
                            color="PURPLE"
                        )

                        await interaction.followup.send(embed=embed)

                    else:
                        embed = generic_colored_embed(
                            title="The song is already paused!",
                            description="",
                            color="ERROR"
                        )

                        await interaction.followup.send(embed=embed)
