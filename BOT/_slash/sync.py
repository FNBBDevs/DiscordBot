import discord
from discord.app_commands import Group
from _utils.embeds import generic_colored_embed


class Sync(Group):
    def __init__(self, tree, guild, args=None):
        @tree.command(
            name="sync", description="Owner only", guild=discord.Object(id=guild)
        )
        async def sync(interaction: discord.Interaction):

            await tree.sync(guild=discord.Object(id=guild))

            embed = generic_colored_embed(
                title="Command Tree Synced",
                description="",
                color="SUCCESS"
            )

            await interaction.followup.send(embed=embed)
