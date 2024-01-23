import os
import discord
from discordwebhook import Discord
from _utils.views import ImagineView


class Imagine:
    """
    Description: adds two numbers.
    """

    def __init__(self, tree, guild, args=None):
        """
        Description: Constructor.
        """

        @tree.command(
            name="imagine",
            description="Creates images with Midjourney",
            guild=discord.Object(id=guild),
        )
        async def imagine(interaction: discord.Interaction, prompt: str):
            """
            /imagine
            """
            
            await interaction.response.defer()

            print(os.getcwd())
         
            with open('./BOT/ignore/a.png', 'rb') as pic:
                imagine_file = discord.File(pic)
            
            await interaction.followup.send(
                f"**{prompt}** - <@{interaction.user.id}> (fast)",
                file=imagine_file,
                view=ImagineView()
            )
