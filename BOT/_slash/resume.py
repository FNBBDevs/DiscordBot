import discord
from discord.app_commands import Group
from _utils.embeds import generic_colored_embed
from _utils.views import PauseView


class Resume(Group):
    def __init__(self, tree, guild, args=None):
        @tree.command(
            description="Resume a song", name="resume", guild=discord.Object(id=guild)
        )
        async def resume(interaction: discord.Interaction):
            """
            Resume a song that was paused.
            """
            # grab the channel the user is in
            user_channel = interaction.user.voice
            # grab the channel the bot is in
            voice_channel = interaction.guild.voice_client
            await interaction.response.defer()
            if user_channel:
                if voice_channel: 
                    if voice_channel.is_paused():
                        voice_channel.resume()
                        embed = generic_colored_embed(
                            title="Success ✅",
                            description="Song has been resumed",
                            color="PURPLE",
                        )
                        await interaction.followup.send(embed=embed, view=PauseView())
                    elif voice_channel.is_playing():
                        embed = generic_colored_embed(
                            title="Error ❌",
                            description="A song is already playing",
                            color="ERROR"
                        )
                        await interaction.followup.send(embed=embed)
                    else:
                        embed = generic_colored_embed(
                            title="Error ❌",
                            description="No song to resume and no song in the queue",
                            color="ERROR"
                        )
                        await interaction.followup.send(embed=embed)
                else:
                    embed = generic_colored_embed(
                        title="Error ❌",
                        description="Not currently in a VC",
                        color="ERROR"
                    )
                    await interaction.followup.send(embed=embed)

