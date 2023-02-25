import discord

class echo:
    def __init__(self, tree, guild):
        @tree.command(name="echo", description="let me talk to you ;)", guild=discord.Object(id=guild))
        async def echo(interaction, message: str = "are you serious right now"):
            """
            /add command
            :param a: number 1
            :param b: number 2
            """
            await interaction.response.send_message(message)