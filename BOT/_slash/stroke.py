import discord
from discord import Interaction 

class stroke:
    def __init__(self, tree, guild):
        @tree.command(name="stroke", description="mmmm yes stroke my big grizzly cock", guild=discord.Object(id=guild))
        async def stroke(interaction: Interaction):
            await interaction.response.defer()
            await interaction.followup.send('<:fortniteballs:1075285121990672435>')
