import discord
from discord import app_commands

class deez:
    TREE = None
    GUILD = None
    def __init__(self, tree, guild):
        TREE = tree
        GUILD = guild

        @TREE.command(name="deez", description="bud is asking for a brusing!!!!! on god!!!!", guild=discord.Object(id=GUILD))
        async def deez(interaction):
            """
            /deez command
            """
            await interaction.response.send_message("nutz [tips hat]")
