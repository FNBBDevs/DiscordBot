"""
DISCORD BOT

Authors: Nolan Gregory, Ethan Christensen, Klim F.
Version: 0.69
Intent: This multifaceted bot is to promote user activity within
        servers that it is hosted in.
"""

import discord
from _commands.contains import Contains
from discord import Message, app_commands
from slash_master import SlashMaster


class FortniteBallsBot(discord.Client):
    """
    Desription: Multipurpose discord bot
    """

    def __init__(self, guild, cmds_path, *, debug, intents: discord.Intents, **options):
        super().__init__(intents=intents, **options)

        # Guild ID the bot will only work in
        self._guild = guild
        self._debug = bool(int(debug))
        self._cmds_path = cmds_path
        self._contains = Contains()
        # Create CommandTree object
        self.tree = app_commands.CommandTree(self)

    async def on_ready(self):
        """
        Description: Called when the bot is initialized (turned on)
        """
        if self._debug:
            print(f"Logged in as: {self.user}")

        # tell slash master to load commands
        SlashMaster(
            self.tree, self._guild, self._cmds_path, self._debug
        ).load_commands()

        # sync the commands with our guild (server)
        # await self.tree.sync(guild=discord.Object(id=self._guild))

    async def on_message(self, message: Message):
        """
        This functions reads incoming messages and replies appropriately
        if the message contains certain flags ;)
        :param message: message from the client
        """
        if message.author == self.user or str(message.channel) in [
            "testing",
            "git-log",
        ]:
            return

        if self._debug:
            print(f"Message from {message.author}: {message.content}")

        if results := self._contains.execute(
            message.content.strip().lower(), self._debug
        ):
            for response in results:
                await message.channel.send(response)

        if "<@1075154837609656350>" in str(message.content):
            await message.channel.send(
                f"Erm... <@{message.author.id}> ... [looks away nervously] ... pwease"
                " don't ping me :("
            )

    async def on_message_delete(self, message):
        """
        Description: Tell when a message was deleted.
        """
        msg = (
            str(message.author)
            + "deleted message in "
            + str(message.channel)
            + ": "
            + str(message.content)
        )
        print(msg)
        # with open("./error.fnbbef", "a+") as f:
        #     f.write(msg + "\n")
