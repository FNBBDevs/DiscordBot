"""
DISCORD BOT

Authors: Nolan Gregory, Ethan Christensen, Klim F.
Version: 0.69
Intent: This multifaceted bot is to promote user activity within
        servers that it is hosted in.
"""

import discord
from discord import app_commands
from slash_master import SlashMaster
from _commands.contains import Contains

class FortniteBallsBot(discord.Client):
    def __init__(self, guild, cmds_path, *, debug, intents: discord.Intents, **options):
        super().__init__(intents=intents, **options)

        # Guild ID the bot will only work in
        self._GUILD = guild
        self._DEBUG = bool(int(debug))
        self._CMDS_PATH = cmds_path
        self._contains = Contains()

    async def on_ready(self):
        if self._DEBUG:
            print(f"Logged in as: {self.user}")

        # Create CommandTree object
        self.tree = app_commands.CommandTree(self)

        # tell slash master to load commands
        SlashMaster(self.tree, self._GUILD, self._CMDS_PATH, self._DEBUG).load_commands()

        # sync the commands with our guild (server)
        await self.tree.sync(guild=discord.Object(id=self._GUILD))

    async def on_message(self, message):
        """
        This functions reads incoming messages and replies appropriately
        if the message contains certain flags ;)
        :param message: message from the client
        """
        if message.author == self.user or str(message.channel) in ['testing', 'git-log']:
            return

        if self._DEBUG: 
            print(f"Message from {message.author}: {message.content}")

        if results := self._contains.execute(message.content.strip().lower(), self._DEBUG):
            for response in results: await message.channel.send(response)

        if "<@1075154837609656350>" in str(message.content):
            await message.channel.send(f"Erm... <@{message.author.id}> ... [looks away nervously] ... pwease don't ping me :(")

    async def on_message_delete(self, message):
        msg = str(message.author)+ 'deleted message in '+str(message.channel)+': '+str(message.content)
        with open("./error.fnbbef", "a+") as f: f.write(msg+"\n")

