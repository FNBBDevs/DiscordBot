import io
import discord

from _utils.embeds import generic_colored_embed

class UpdateAvatar:
    """
    Description: adds two numbers.
    """

    def __init__(self, tree, guild, args=None):
        """
        Description: Constructor.
        """
        
        @tree.command(
            name="update_avatar",
            description="Change the bots avatar!",
            guild=discord.Object(id=guild),
        )
        async def update_avatar(interaction: discord.Interaction, image: discord.Attachment):
            """
            /update_avatar command
            :param image: new avatar image
            """
            if interaction.user.global_name != "etchris":
                await interaction.response.send_message(embed=generic_colored_embed(
                        title="Avatar Update",
                        description=f"An error was encountered updating the bots avatar!",
                        footer_text="Requested by:",
                        footer_img=interaction.user.avatar,
                        footer_usr=interaction.user.global_name,
                        color="ERROR"
                    ).add_field(name="Error", value=f"User {interaction.user.global_name} does not have permission to update the avatar!", inline=False).set_thumbnail(
                        url="https://cdn.discordapp.com/attachments/1194024444574843004/1208973861207408710/lol-rofl.gif?ex=65e53ba6&is=65d2c6a6&hm=c3870849ad6c05a20961ce28264954859d55ce808d81906bfe7ed3c338073a7b&"
                    ))
                return
            
            await interaction.response.defer()
            
            if image.content_type not in ['image/png', 'image/jpeg']:
                await interaction.followup.send(
                    embed=generic_colored_embed(
                        title="Avatar Update",
                        description=f"An error was encountered updating the bots avatar!",
                        footer_text="Requested by:",
                        footer_img=interaction.user.avatar,
                        footer_usr=interaction.user.global_name,
                        color="ERROR"
                    ).add_field(name="Error", value=f"The file type must be of `.png` or `.jpg`, not {image.content_type}.", inline=False)
                )
            else:
                try:
                    image_bytes = await image.read()
                    image_file = discord.File(io.BytesIO(image_bytes), filename="avatar.png")
                    await interaction.client.user.edit(avatar=image_bytes)
                    embed = generic_colored_embed(
                        title="Avatar Update",
                        description="The bots avatar has been sucessfully updated!",
                        footer_text="Requested by:",
                        footer_img=interaction.user.avatar,
                        footer_usr=interaction.user.global_name,
                        color="SUCCESS"
                    )
                    embed.set_image(url="attachment://avatar.png")
                    await interaction.followup.send(
                        embed=embed,
                        file=image_file
                    )
                except Exception as e:
                    await interaction.followup.send(
                        embed=generic_colored_embed(
                            title="Avatar Update",
                            description=f"An error was encountered updating the bots avatar!",
                            footer_text="Requested by:",
                            footer_img=interaction.user.avatar,
                            footer_usr=interaction.user.global_name,
                            color="ERROR"
                        ).add_field(name="Error", value=str(e), inline=False)
                    )
                