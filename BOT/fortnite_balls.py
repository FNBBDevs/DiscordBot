"""
DISCORD BOT

Authors: Nolan Gregory, Ethan Christensen, Klim F.
Version: 0.69
Intent: This multifaceted bot is to promote user activity within
        servers that it is hosted in.
"""

import os
import discord
import asyncio
from _commands.contains import Contains
from _utils.embeds import generic_colored_embed
from _utils.queueing import MusicQueue
from discord import Message, app_commands
from discord.ext import commands
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
            "music_queue": MusicQueue()
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
        # subprocess.Popen(["python", "./BOT/_utils/boot.py"], close_fds=True)

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
    
    @commands.Cog.listener()
    async def on_voice_state_update(self, member, before, after):
        try:
            # get the client to get channels, globals
            client = self.get_guild(int(self._guild))

            # get the info channel to send embeds
            info_channel = [channel for channel in client.channels if str(channel.id) == str(os.getenv("INFO_CHANNEL"))][0]

            # remove records from queue that user who left requested (like a boss)
            purged_records = self._fnbb_globals.get("music_queue").check_and_purge(member.name)

            if purged_records > 0:
                purged_embed = generic_colored_embed(
                    title=f"{member.name} has left VC",
                    description=f"Songs by {member.name} purged from the Queue: {purged_records}",
                    color="ORANGE"
                )

                await info_channel.send(embed=purged_embed)
            
            # check to see if the bot is the only one in VC
            members = client.voice_client.channel.members

            # give a minute warning
            if len(members) <= 1:
                timeout_embed = generic_colored_embed(
                    title="Bot Leaving VC",
                    description="In 1 minute the bot will leave VC unless someone joins.",
                )
                await info_channel.send(embed=timeout_embed)
                await asyncio.sleep(60)
                # check again and leave if bot is the only one in vc
                if len(members) <= 1:
                    await self.get_guild(int(self._guild)).voice_client.disconnect()
        except Exception as e:
            pass
