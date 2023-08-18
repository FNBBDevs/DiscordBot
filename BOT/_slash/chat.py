
import discord
from _utils.modals import OpenAIPasswordInputModal


class Chat:
    """
    Description: adds two numbers.
    """

    def __init__(self, tree, guild, args=None):
        """
        Description: Constructor.
        """

        @tree.command(
            name="chat",
            description="chat with Marcus!",
            guild=discord.Object(id=guild),
        )
        async def chat(interaction, message: str):
            """
            /add command
            :param a: message for Marcus
            :param b: password to utilize the service
            """

            modal = OpenAIPasswordInputModal(prompt=message, title="Chat Password")

            await interaction.response.send_modal(modal)
