import _utils.embeds as embeds
import discord
from discord.app_commands import Group


class Resume(Group):
    def __init__(self, tree, guild, args=None):
        @tree.command(
            description="Resume a song", name="resume", guild=discord.Object(id=guild)
        )
        async def resume(interaction: discord.Interaction):
            """
            Resume song from queue.
            """

            # INITIALIZE USER STATUS AND BOT STATUS
            user_channel = interaction.user.voice
            voice_channel = interaction.guild.voice_client

            # AWAIT A RESPONSE
            await interaction.response.defer()

            if user_channel:
                # SEE IF THE BOT IS IN THE CHANNEL
                if voice_channel:
                    try:
                        voice_channel.resume()
                        await interaction.followup.send(
                            content="Resumed!"
                        )
                    except Exception as e:
                        print(e)
