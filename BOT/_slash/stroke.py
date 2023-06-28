import discord
from discord import Interaction


class Stroke:
    """
    Description: Really klim
    """

    def __init__(self, tree, guild):
        """
        Description: Really klim
        """

        @tree.command(
            name="stroke",
            description="mmmm yes stroke my big grizzly cock",
            guild=discord.Object(id=guild),
        )
        async def stroke(interaction: Interaction):
            """
            Description: Really klim
            """
            await interaction.response.defer()
            await interaction.followup.send("<:fortniteballs:1075285121990672435>")

