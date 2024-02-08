import discord
from discord.app_commands import Group
from _utils.embeds import generic_colored_embed

# Bro this needs some serious help. A fix will need to be made to this whole command
class Queue(Group):
    def __init__(self, tree, guild, args=None):
        @tree.command(
            description="Show what is currently in the queue.",
            name="queue",
            guild=discord.Object(id=guild),
        )
        async def queue(interaction: discord.Interaction):

            await interaction.response.defer()

            current_queue = interaction.client._fnbb_globals.get("music_queue")
            
            if len(current_queue) != 0:
                embed = generic_colored_embed(
                    title=f"Queue",
                    description=f"There are currently {len(current_queue)} song{'s' if len(current_queue) > 1 else ''} in the Queue",
                    footer_img=interaction.user.guild_avatar,
                    footer_text="",
                    footer_usr=interaction.user.display_name
                )
                
                for idx, song in enumerate(current_queue.queue):
                    embed.add_field(
                        name=f"Song {idx + 1}",
                        value=f"{song.url} - {song.audio_filter} - {song.user}",
                        inline=False
                    )
                
                await interaction.followup.send(
                    embed=embed
                )

            else:
                await interaction.followup.send(
                    embed = generic_colored_embed(
                        title="The Queue is empty",
                        description="There are currently no songs in the Queue",
                        footer_img=interaction.user.guild_avatar,
                        footer_text="",
                        footer_usr=interaction.user.display_name
                    )
                )