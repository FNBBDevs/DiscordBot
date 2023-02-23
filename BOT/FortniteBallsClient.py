
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

CONTAINS       = contains.Contains()
BOT_NAME       = "Fortnite Balls Bot"
DEBUG          = True
#commands       = COMMANDS.COMMANDS <- deprecated???
#COMMAND_PREFIX = '!' <--------------- deprecated???

class FortniteClient(discord.Client):

    # need this to register this clients on_message command tnx
    async def on_ready():
        pass

    async def on_message(self, message):
        """
        This functions reads incoming messages and replies appropriatley 
        if the message contains certain flags ;)
        :param message: message from the client
        """
        if message.author == self.user:return

        if DEBUG: print(f"Message from {message.author}: {message.content}")

        if results := CONTAINS.execute(message.content.strip().lower(), DEBUG):
            for response in results: await message.channel.send(response)
        
        if str(message.content).__contains__("<@1075154837609656350>"):
            await message.channel.send(f"Hey, <@{message.author.id}>, don't ping me, bud!")
       

        # ETCHRIS SAYS - vvvvv deprecated now that slash works???? vvvvv
        """ # Let's assume we want to parse bot commands - we could simply do something like what I illustrate below.
        if message.content[0] == COMMAND_PREFIX:
            # Here we could go ahead and search through a command base - I will just code one in as an example.
            if not message.content in commands:
                await message.channel.send(f"Hey, <@{message.author.id}>, that command was not found, bud!")
            else:
                await message.channel.send(commands[message.content])"""
