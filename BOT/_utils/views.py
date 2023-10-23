import discord
import datetime
from .embeds import generic_colored_embed

class PauseView(discord.ui.View):
    @discord.ui.button(label="Pause", style=discord.ButtonStyle.primary, emoji="⏸️")
    async def button_callbak(self, interaction, button):
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
    async def button_callbak(self, interaction, button):
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

        await interaction.followup.send(embed=embed, view=PauseView())
