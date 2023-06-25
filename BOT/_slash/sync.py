import discord
from discord.app_commands import Group
from discord.app_commands import CommandTree

class sync(Group):
    def __init__(self, tree, guild):
        @tree.command(name='sync', description='Owner only', guild=discord.Object(id=guild))
        async def sync(interaction: discord.Interaction):
            await tree.sync(guild=discord.Object(id=self._GUILD))
            await interaction.response.send_message('Command tree synced.')
        