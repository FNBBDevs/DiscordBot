import discord
from discord.app_commands import Group


class Kick(Group):
    def __init__(self, tree, guild):
        @tree.command(
            description="Kick the bot from VC",
            name="disconnect",
            guild=discord.Object(id=guild),
        )
        async def kick(interaction: discord.Interaction):
            """
            Force bot to leave the voice channel.
            """
            try:
                # Checks to see if a user is in a voice channel
                bot_channel = interaction.guild.voice_client
                if bot_channel:
                    try:
                        # Safely cleanup and disconnect the bot
                        bot_channel.cleanup()
                        await bot_channel.disconnect()
                        return

                    except Exception as e:
                        # If user is not in the voice channel
                        await interaction.response.send_message(f"{e}")
                else:
                    await interaction.response.send_message(
                        "Bruh I'm literally not in a voice channel rn..."
                    )
            except Exception as e:
                # If user is not in the voice channel
                await interaction.response.send_message(f"{e}")
