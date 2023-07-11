import discord
from discord.app_commands import Group


class Sync(Group):
    def __init__(self, tree, guild):
        @tree.command(
            name="sync", description="Owner only", guild=discord.Object(id=guild)
        )
        async def sync(interaction: discord.Interaction):
            # Sync that command tree up!
            await tree.sync(guild=discord.Object(id=guild))
            await interaction.response.send_message("Command tree synced.")
