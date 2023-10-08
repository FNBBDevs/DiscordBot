import discord


class Add:
    """
    Description: adds two numbers.
    """

    def __init__(self, tree, guild, args=None):
        """
        Description: Constructor.
        """

        @tree.command(
            name="add",
            description="balls, adding two numbers",
            guild=discord.Object(id=guild),
        )
        async def add(interaction, int_one: int = 60, int_two: int = 9):
            """
            /add command
            :param a: number 1
            :param b: number 2
            """
            await interaction.response.send_message(
                f"{int_one} + {int_two} = {int_one+int_two}"
            )
