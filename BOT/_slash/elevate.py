import discord


class Elevate:
    """
    Description: adds two numbers.
    """

    def __init__(self, tree, guild, args=None):
        """
        Description: Constructor.
        """

        @tree.command(
            name="elevate",
            description="asdf",
            guild=discord.Object(id=guild),
        )
        async def elevate(interaction: discord.Interaction, user: discord.User, role: discord.Role):
            
            if interaction.user.id in [933796468731568191,411399698679595008]:
                member = interaction.guild.get_member(user.id)
                if role.position > interaction.guild.me.top_role.position:
                    await interaction.response.send_message(f"I don't have the permissions to give this role.")
                else:
                    await member.add_roles(role)
                    await interaction.response.send_message(f"Role {role.name} has been given to {user.name}.")
            else:
                interaction.response.send_message("No permission!")
