import discord
from discord.app_commands import Group
from discord.app_commands import CommandTree


class Dm(Group):
    def __init__(self, tree: CommandTree, guild):
        super().__init__(name="dm")

        @self.command(description="Send a direct message to a specific user")
        async def user(
            interaction: discord.Interaction, member: discord.Member, message: str
        ):
            # Create DMs if they don't already exist
            dms: discord.DMChannel = await member.create_dm()

            await dms.send(message)
            await interaction.response.send_message(
                content=f"Sent a DM to {member.mention}\n>>> {message}", ephemeral=True
            )

        @self.command(
            description="Send a direct message to everyone in the discord server"
        )
        async def all(interaction: discord.Interaction, message: str):
            await interaction.response.defer(ephemeral=True, thinking=True)

            async for member in interaction.guild.fetch_members():
                if member.bot:
                    continue

                try:
                    # Create DMs if they don't already exist
                    dms: discord.DMChannel = await member.create_dm()
                    await dms.send(message)

                except discord.errors.Forbidden as e:
                    print(f"Could not direct message {member.name}\nReason: {e}")

            await interaction.followup.send(
                content=f"Sent a DM to everyone\n>>> {message}", ephemeral=True
            )

        tree.add_command(self, guild=discord.Object(id=guild))
