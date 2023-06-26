import discord
from discord.app_commands import Group
from discord.app_commands import CommandTree

# Bro this needs some serious help. A fix will need to be made to this whole command
class playing(Group):
    def __init__(self, tree, guild):
        @tree.command(description="Show what song is currently playing", name="playing", guild=discord.Object(id=guild))
        async def playing(interaction : discord.Interaction):
            """
            Shows what song is playing.
            """

            user_channel = interaction.user.voice
            await interaction.response.defer()

            # IF THE BOT IS NOT THE VOICE CHANNEL, ADD TO CHANNEL
            if not user_channel:
                await interaction.followup.send("NIGGA I AINT EVEN *IN* A VC RIGHT NOW! :skull: :rofl:")
            # SEE IF THE BOT IS IN THE CHANNEL - WE DON'T NEED TO JOIN
            elif interaction.guild.voice_client:
                voice_channel = interaction.guild.voice_client
                if voice_channel:
                    if voice_channel.channel:
                        channel = interaction.guild.voice_client.channel
                        embed = discord.Embed(title=f"Currently Playing", description=interaction.guild.voice_client.channel, color= 0xB6CDE4)
                        #embed.set_thumbnail(url="https://assets.stickpng.com/images/580b57fcd9996e24bc43c4bf.png")
                        embed.set_footer(text=f'Requested By: {interaction.user.name} (fag)', icon_url= interaction.user.guild_avatar)
                        await interaction.followup.send(embed=embed)
                    else:
                        await interaction.followup.send("NIGGA I AINT EVEN *PLAYING MUSIC*! :skull: :rofl:")