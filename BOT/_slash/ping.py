import discord
from discord.app_commands import Group
from discord.app_commands import CommandTree


class Ping(Group):
    def __init__(self, tree: CommandTree, guild):
        super().__init__(name="ping")

        @self.command(description="Ghost ping a user N amount of times.")
        async def ghost(
            interaction: discord.Interaction, member: discord.Member, amount: int
        ):
            """
            Ghost ping in which the message is deleted from the channel.
            """
            if (amount < 0) or (amount > 15):
                await interaction.response.send_message(
                    content="Enter integer between 1-15.", ephemeral=True
                )
            else:
                await interaction.response.send_message(
                    content=f"Ghost pinging {member.name} {amount} times.",
                    ephemeral=True,
                )

                for i in range(0, amount):
                    await interaction.channel.send(f"{member.mention}")
                    await interaction.channel.last_message.delete()

        @self.command(
            description="Ping a user N amount of times with a custom message."
        )
        async def perpetual(
            interaction: discord.Interaction,
            member: discord.Member,
            amount: int,
            message: str,
        ):
            """
            Perpetual ping in which the message remains in the channel.
            """
            if (amount < 0) or (amount > 15):
                await interaction.response.send_message(
                    content="Enter integer between 1-15.", ephemeral=True
                )
            else:
                await interaction.response.send_message(
                    content=(
                        f"Sending {member.name} the message {message} {amount} times."
                    ),
                    ephemeral=True,
                )

                for i in range(0, amount):
                    await interaction.channel.send(f"{member.mention} {message}")

        tree.add_command(self, guild=discord.Object(id=guild))
