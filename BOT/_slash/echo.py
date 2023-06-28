import discord


class echo:
    def __init__(self, tree, guild):
        @tree.command(
            name="echo",
            description="let me talk to you ;)",
            guild=discord.Object(id=guild),
        )
        async def echo(interaction, message: str = "are you serious right now"):
            """
            Echo command
            """
            await interaction.response.send_message(message)
