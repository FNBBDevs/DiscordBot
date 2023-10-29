import _utils.embeds as embeds
import discord
from discord.app_commands import Group


class Skip(Group):
    def __init__(self, tree, guild, args=None):
        @tree.command(
            description="Skip a song", name="skip", guild=discord.Object(id=guild)
        )
        async def skip(interaction: discord.Interaction):
            """
            skip song from queue.
            """

            # INITIALIZE USER STATUS AND BOT STATUS
            user_channel = interaction.user.voice
            voice_channel = interaction.guild.voice_client

            # AWAIT A RESPONSE
            await interaction.response.defer()

            # IF THE USER IS NOT THE VOICE CHANNEL, ISSUE ERROR
            if not user_channel:
                await interaction.followup.send(
                    embed=embeds.generic_colored_embed(
                        title="You are Not In a Voice Channel!",
                        description=(
                            "Hermph... You must be in a voice channel to skip songs."
                        ),
                        footer_text="(Attempted) Skip By:",
                        footer_usr=interaction.user.name,
                        footer_img=interaction.user.guild_avatar,
                        color="ERROR"
                    )
                )

            # SEE IF THE BOT IS IN THE CHANNEL
            elif voice_channel:
                # If the bot is currently playing music
                if voice_channel.is_playing():
                    # This is BROKEN
                    voice_channel.stop()

                    await interaction.followup.send(
                        embed=embeds.generic_colored_embed(
                            title="Successfully Skipped Song!",
                            description="",
                            footer_text="Skipped By:",
                            footer_usr=interaction.user.name,
                            footer_img=interaction.user.guild_avatar,
                            color="SUCCESS"
                        )
                    )
                # If the bot is in the voice channel but no music is playing
                else:
                    await interaction.followup.send(
                        embed=embeds.generic_colored_embed(
                            title="No Song to Skip!",
                            description=(
                                "Erm... The queue is empty and there is no song to"
                                " skip..."
                            ),
                            footer_text="(Attempted) Skip By:",
                            footer_usr=interaction.user.name,
                            footer_img=interaction.user.guild_avatar,
                            color="WARNING"
                        )
                    )

            # BOT IS NOT IN THE CHANNEL
            else:
                await interaction.followup.send(
                    embed=embeds.generic_colored_embed(
                        title="Bot Not In Voice Channel!",
                        description=(
                            "Hermph... I must be in a voice channel to skip songs..."
                        ),
                        footer_text="(Attempted) Skip By:",
                        footer_usr=interaction.user.name,
                        footer_img=interaction.user.guild_avatar,
                        color="ERROR"
                    )
                )
