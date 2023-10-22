import discord
from discord.app_commands import Group
from _utils.embeds import generic_colored_embed


class Resume(Group):
    def __init__(self, tree, guild, args=None):
        @tree.command(
            description="Resume a song", name="resume", guild=discord.Object(id=guild)
        )
        async def resume(interaction: discord.Interaction):
            """
            Resume song from queue.
            """

            user_channel = interaction.user.voice
            
            voice_channel = interaction.guild.voice_client

            await interaction.response.defer()

            if user_channel:
                if voice_channel:
                    if not voice_channel.is_playing():

                        voice_channel.resume()

                        embed = generic_colored_embed(
                            title="Song has been resumed!",
                            description="",
                            color="PURPLE"
                        )

                        await interaction.followup.send(embed=embed)

                    else:
                        embed = generic_colored_embed(
                            title="The song is already playing!",
                            description="",
                            color="ERROR"
                        )

                        await interaction.followup.send(embed=embed)
