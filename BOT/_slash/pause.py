import _utils.embeds as embeds
import discord
from discord.app_commands import Group


class Pause(Group):
    def __init__(self, tree, guild, args=None):
        @tree.command(
            description="Pause a song", name="pause", guild=discord.Object(id=guild)
        )
        async def pause(interaction: discord.Interaction):
            """
            Pause song from queue.
            """

            # INITIALIZE USER STATUS AND BOT STATUS
            user_channel = interaction.user.voice
            voice_channel = interaction.guild.voice_client

            # AWAIT A RESPONSE
            await interaction.response.defer()
            if user_channel:
                # SEE IF THE BOT IS IN THE CHANNEL
                if voice_channel:
                    # If the bot is currently playing music
                    if voice_channel.is_playing():
                        # This is BROKEN
                        voice_channel.pause()
                        await interaction.followup.send(
                            content="Paused!"
                        )
