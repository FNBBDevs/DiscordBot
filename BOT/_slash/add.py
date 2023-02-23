import discord
from discord import app_commands

class add:
    TREE = None
    GUILD = None
    def __init__(self, tree, guild):
        TREE = tree
        GUILD = guild
    
        @TREE.command(name="add", description="balls, adding two numbers", guild=discord.Object(id=GUILD))
        async def add(interaction, a: int = 60, b: int = 9):
            """
            /add command
            :param a: number 1
            :param b: number 2
            """
            await interaction.response.send_message(f"{a} + {b} = {a+b}")
