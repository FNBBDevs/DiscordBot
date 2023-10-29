import discord
import datetime
from .embeds import generic_colored_embed

class PauseView(discord.ui.View):
    @discord.ui.button(label="Pause", style=discord.ButtonStyle.primary, emoji="⏸️")
    async def pause_button_callback(self, interaction, button):
        await interaction.response.defer()

        embed = generic_colored_embed(
            title="Success ✅",
            description="Song has been paused",
            color="PURPLE",
        )

        interaction.guild.voice_client.pause()

        original_response = await interaction.original_response()

        await original_response.edit(embed=generic_colored_embed(
            title="Success ✅",
            description="Song has been resumed",
            color="PURPLE",
        ), view=None)

        await interaction.followup.send(embed=embed, view=ResumeView())

class ResumeView(discord.ui.View):
    @discord.ui.button(label="Resume", style=discord.ButtonStyle.primary, emoji="▶️")
    async def resume_button_callback(self, interaction, button):
        await interaction.response.defer()
        embed = generic_colored_embed(
            title="Success ✅",
            description="Song has been resumed",
            color="PURPLE",
        )
        interaction.guild.voice_client.resume()

        original_response = await interaction.original_response()

        await original_response.edit(embed=generic_colored_embed(
            title="Success ✅",
            description="Song has been paused",
            color="PURPLE",
        ), view=None)

        await interaction.followup.send(embed=embed)
    
class PlayingView(discord.ui.View):
    @discord.ui.button(label="Pause", style=discord.ButtonStyle.primary, emoji="⏸️")
    async def pause_button_callback(self, interaction, button):
        await interaction.response.defer()

        embed = generic_colored_embed(
            title="Success ✅",
            description="Song has been paused",
            color="PURPLE",
        )

        interaction.guild.voice_client.pause()

        original_response = await interaction.original_response()

        await original_response.edit(embed=original_response.embeds[0], view=None)

        await interaction.followup.send(embed=generic_colored_embed(
            title="Success ✅",
            description="Song has been paused",
            color="PURPLE",
        ), view=ResumeView())
    
    @discord.ui.button(label="Skip", style=discord.ButtonStyle.danger, emoji="⏭️")
    async def skip_button_callback(self, interaction, button):
            user_channel = interaction.user.voice
            voice_channel = interaction.guild.voice_client

            # AWAIT A RESPONSE
            await interaction.response.defer()

            original_response = await interaction.original_response()

            await original_response.edit(embed=original_response.embeds[0], view=None)

            # IF THE USER IS NOT THE VOICE CHANNEL, ISSUE ERROR
            if not user_channel:
                await interaction.followup.send(
                    embed=generic_colored_embed(
                        title="You are Not In a Voice Channel!",
                        description=(
                            "Hermph... ❌"
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
                        embed=generic_colored_embed(
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
                        embed=generic_colored_embed(
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
                    embed=generic_colored_embed(
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


