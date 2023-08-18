import discord


class Echo:
    def __init__(self, tree: discord.app_commands.CommandTree, guild: str, args=None):
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
