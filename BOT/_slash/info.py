import random

import discord
from discord.app_commands import CommandTree, Group


class Info(Group):
    def __init__(self, tree: CommandTree, guild: str, args=None):
        super().__init__(name="info")

        @self.command(description="Display information about the discord server")
        async def server(interaction: discord.Interaction):
            embed = discord.Embed(
                title=interaction.guild.name,
                description=f"Members ({interaction.guild.member_count})",
                url=interaction.guild.icon.url,
                color=random.randint(0, 0xFFFFFF),
            )

            embed.add_field(name="Owner", value=interaction.guild.owner.display_name)
            embed.set_image(url=interaction.guild.icon.url)
            embed.set_footer(
                text=f'Created {interaction.guild.created_at.strftime("%B %d, %Y")}',
                icon_url="https://bit.ly/3SJZQwM",
            )

            await interaction.response.send_message(embed=embed)

        @self.command(
            description=(
                "Display information about a specific user or yourself by default"
            )
        )
        async def user(
            interaction: discord.Interaction,
            user: discord.User = None,
            show_display_info: bool = False,
        ):
            if not user:
                user = interaction.user

            avatar = user.display_avatar if show_display_info else user.avatar
            name = user.display_name if show_display_info else user.name
            desc = f'(AKA "{user.name}")' if show_display_info else "who dis?"

            embed = discord.Embed(
                title=name, description=desc, url=avatar.url, color=user.color
            )
            embed.set_image(url=avatar.url)
            embed.set_footer(
                text=f'Created {user.created_at.strftime("%B %d, %Y")}',
                icon_url="https://bit.ly/3SJZQwM",
            )

            await interaction.response.send_message(embed=embed)

        tree.add_command(self, guild=discord.Object(id=guild))
