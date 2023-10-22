"""
DISCORD BOT

Authors: Nolan Gregory, Ethan Christensen, Klim F.
Version: 0.69
Intent: This multifaceted bot is to promote user activity within
        servers that it is hosted in.
"""

import re
import subprocess
import discord
from _commands.contains import Contains
from _utils.bruhpy import BruhPy
from _utils.embeds import bruhby, nolang
from _utils.nolang import Nolang
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
        self._fnbb_globals = {
            "playing": {},
        }
        # Create CommandTree object
        self.tree = app_commands.CommandTree(self)

    async def on_ready(self):
        """
        Description: Called when the bot is initialized (turned on)
        """

        # get the text channels in the guild
        guild = self.get_guild(int(self._guild))
        text_channels = guild.text_channels
        
        # call the cool animation while the commands load
        subprocess.Popen(["python", "./BOT/_utils/boot.py"], close_fds=True)

        # tell slash master to load commands
        SlashMaster(self.tree, self._guild, self._cmds_path, self._debug).load_commands(
            args=(text_channels,)
        )

    async def on_message(self, message: Message):
        """
        This functions reads incoming messages and replies appropriately
        if the message contains certain flags ;)
        :param message: message from the client
        """
        if message.author == self.user:
            return

        contains_results = self._contains.execute(
            message.content.strip().lower(), self._debug
        )

        if contains_results and str(message.channel) not in ["testing", "git-log"]:
            for response in contains_results:
                await message.channel.send(response)

        if "<@1075154837609656350>" in str(message.content):
            await message.channel.send(
                f"Erm... <@{message.author.id}> ... [looks away nervously] ... pwease"
                " don't ping me :("
            )

        if message.content[0:8] == ":python:":
            code = message.content[8:].lstrip().rstrip()
            response = BruhPy().run(code, "", message.author.display_name)
            embed_response = bruhby(response[0], message.author.display_name)
            await message.channel.send(embed=embed_response)
            del response
            del embed_response
        elif x := re.search(r"\<\:python\:\d*\>", message.content):
            start, end = x.span()
            code = message.content[end:].lstrip().rstrip()
            response = BruhPy().run(code, "", message.author.display_name)
            embed_response = bruhby(response[0], message.author.display_name)
            await message.channel.send(embed=embed_response)
            del response
            del embed_response

        if message.content[0:8] == ":nolang:":
            code = message.content[8:].lstrip().rstrip()
            response = Nolang().run(code, "")
            embed_response = nolang(response[0], message.author.display_name)
            await message.channel.send(embed=embed_response)
            del response
            del embed_response
        elif x := re.search(r"\<\:nolang\:\d*\>", message.content):
            start, end = x.span()
            code = message.content[end:].lstrip().rstrip()
            response = Nolang().run(code, "")
            embed_response = nolang(response[0], message.author.display_name)
            await message.channel.send(embed=embed_response)
            del response
            del embed_response

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
