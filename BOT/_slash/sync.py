import discord
from discord.app_commands import Group
from _utils.embeds import generic_colored_embed


class Sync(Group):
    def __init__(self, tree, guild, args=None):
        @tree.command(
            name="sync", description="Owner only", guild=discord.Object(id=guild)
        )
        async def sync(interaction: discord.Interaction):
            await interaction.response.defer(ephemeral=True)
            try:
                await tree.sync(guild=discord.Object(id=guild))

                embed = generic_colored_embed(
                    title="Command Tree Synced",
                    description="",
                    color="SUCCESS"
                )
            except Exception as e:
                embed = generic_colored_embed(
                    title="Command Tree failed to Sync",
                    description=str(e),
                    color="ERROR"
                )

            await interaction.followup.send(embed=embed, ephemeral=True)
