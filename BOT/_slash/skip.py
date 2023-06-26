import discord
from discord.app_commands import Group
from discord.app_commands import CommandTree
import _utils.embeds as embeds

class skip(Group):
    def __init__(self, tree, guild):
        @tree.command(description="Skip a song", name="skip", guild=discord.Object(id=guild))
        async def skip(interaction : discord.Interaction):
            """
            skip song from queue.
            """

            # INITIALIZE USER STATUS AND BOT STATUS
            user_channel = interaction.user.voice
            voice_channel = interaction.guild.voice_client

            # AWAIT A RESPONSE
            await interaction.response.defer()
            print("channel" + str(type(voice_channel)))
            print("voice_client" + str(type(interaction.guild.voice_client)))

            # IF THE USER IS NOT THE VOICE CHANNEL, ISSUE ERROR
            if not user_channel:
                await interaction.followup.send(embed=embeds.on_error(title="You are Not In a Voice Channel!",
                                                                      description="Hermph... You must be in a voice channel to skip songs.",
                                                                      footer_text="(Attempted) Skip By:",
                                                                      footer_usr=interaction.user.name,
                                                                      footer_img=interaction.user.guild_avatar))
            
            # SEE IF THE BOT IS IN THE CHANNEL
            elif voice_channel:
                # If the bot is currently playing music
                if voice_channel.is_playing():
                    voice_channel.stop()
                    embed = discord.Embed(title=f"Skipped song", description="Your song sucked and I did not want to listen to it anymore...", color= 0x28a745)
                    embed.set_footer(text=f'Skipped By: {interaction.user.name}', icon_url= interaction.user.guild_avatar)
                    await interaction.followup.send(embed=embeds.on_success(title="Skipped Song!",
                                                                      description="Your song sucked and I did not want to listen to it anymore...",
                                                                      footer_text="Skipped By:",
                                                                      footer_usr=interaction.user.name,
                                                                      footer_img=interaction.user.guild_avatar))
                # If the bot is in the voice channel but no music is playing
                else:
                    await interaction.followup.send(embed=embeds.on_warning(title="No Song to Skip!",
                                                                      description="Erm... The queue is empty and there is no song to skip...",
                                                                      footer_text="(Attempted) Skip By:",
                                                                      footer_usr=interaction.user.name,
                                                                      footer_img=interaction.user.guild_avatar))

            # BOT IS NOT IN THE CHANNEL
            else:
                await interaction.followup.send(embed=embeds.on_error(title="Bot Not In Voice Channel!",
                                                                      description="Hermph... I must be in a voice channel to skip songs...",
                                                                      footer_text="(Attempted) Skip By:",
                                                                      footer_usr=interaction.user.name,
                                                                      footer_img=interaction.user.guild_avatar))
                    