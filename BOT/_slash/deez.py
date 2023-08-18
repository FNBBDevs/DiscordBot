import discord


class Deez:
    def __init__(self, tree, guild, args=None):
        @tree.command(
            name="deez",
            description="bud is asking for a brusing!!!!! on god!!!!",
            guild=discord.Object(id=guild),
        )
        async def deez(interaction):
            """
            /deez command
            """
            await interaction.response.send_message("nutz [tips hat]")
