
"""
DISCORD BOT

Authors: Nolan Gregory, Ethan Christensen , Klim F.
Version: 0.69
Intent: This multifaceted bot is to promote user activity within
        servers that it is hosted in.
"""


# import COMMANDS
import discord
from _commands import contains
import random
import asyncio

CONTAINS       = contains.Contains()
BOT_NAME       = "Fortnite Balls Bot"
DEBUG          = True
USER_ID        = "<@760901705528508526>" #User to ghost ping
#commands       = COMMANDS.COMMANDS <- deprecated??? -- Correct
#COMMAND_PREFIX = '!' <--------------- deprecated??? -- Correct

class FortniteClient(discord.Client):

    async def ghost_ping(self : discord.Client):
        while True:
            channel: discord.TextChannel = random.choice(self.get_guild(1069835760859107368).text_channels)
            msg: discord.Message = await channel.send('<@760901705528508526>')
            await msg.delete()
            await asyncio.sleep(random.randint(3 * 60, 3 * 3600))
            msg: discord.Message = await channel.send(f'{USER_ID}')
            await msg.delete()

    # need this to register this clients on_message command tnx
    async def on_ready(self):
        print(f"Logged in as: {self.user}")
        await self.ghost_ping()

    async def on_message(self, message):
        """
        This functions reads incoming messages and replies appropriately
        if the message contains certain flags ;)
        :param message: message from the client
        """
        if message.author == self.user:return

        if DEBUG: print(f"Message from {message.author}: {message.content}")

        if results := CONTAINS.execute(message.content.strip().lower(), DEBUG):
            for response in results: await message.channel.send(response)

        if "<@1075154837609656350>" in str(message.content):
            await message.channel.send(f"Erm... <@{message.author.id}> ... [looks away nervously] ... pwease don't ping me :(")

        # NULZO SAYS - vvvvv Refactored this for clarity vvvvv
        # if str(message.content).__contains__("<@1075154837609656350>"):
        #     await message.channel.send(f"Hey, <@{message.author.id}>, don't ping me, bud!")
       

        # ETCHRIS SAYS - vvvvv deprecated now that slash works???? vvvvv
        # NULZO SAYS - Correct. This is now deprecated and can be removed on the next version.
        """ # Let's assume we want to parse bot commands - we could simply do something like what I illustrate below.
        if message.content[0] == COMMAND_PREFIX:
            # Here we could go ahead and search through a command base - I will just code one in as an example.
            if not message.content in commands:
                await message.channel.send(f"Hey, <@{message.author.id}>, that command was not found, bud!")
            else:
                await message.channel.send(commands[message.content])"""
