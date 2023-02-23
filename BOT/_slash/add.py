import discord

class add:
    def __init__(self, tree, guild):
        @tree.command(name="add", description="balls, adding two numbers", guild=discord.Object(id=guild))
        async def add(interaction, a: int = 60, b: int = 9):
            """
            /add command
            :param a: number 1
            :param b: number 2
            """
            await interaction.response.send_message(f"{a} + {b} = {a+b}")
