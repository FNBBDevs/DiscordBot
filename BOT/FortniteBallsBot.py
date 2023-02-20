#
# DISCORD BOT

# Authors: Nolan Gregory, ... , ...
# Version: 0.69
# Intent: This multifaceted bot is to promote user activity within
#         servers that it is hosted in.
#

# --------------------------- IMPORT STATEMENTS -------------------------------

import asyncio
import Config
import re
import discord
import COMMANDS
import FUNCTIONS
import random

# ------------------------- VARIABLE INSTANTIATION ----------------------------

COMMAND_PREFIX = '!'
BOT_NAME = "Fortnite Balls Bot"
intents = discord.Intents.default()
intents.message_content = True
commands = COMMANDS.COMMANDS

# --------------------------- CLASS DECLARATION --------------------------------


class FortniteBot(discord.Client):

    async def ghost_ping(self : discord.Client, user_id : int, channel : discord.TextChannel):
        msg: discord.Message = await channel.send(f'<@{user_id}>')
        await msg.delete()

    async def ping_joe(self : discord.Client):
        while True:
            channel: discord.TextChannel = random.choice(self.get_guild(1069835760859107368).text_channels)
            await self.ghost_ping(760901705528508526, channel)
            
            # DBG: Ghost ping spice for testing purposes
            await self.ghost_ping(405912283554906123, channel)

            await channel.send('https://cdn.discordapp.com/attachments/874659780764004375/1057152226407227402/VID_276710815_003119_573.mp4')
            await asyncio.sleep(random.randint(3 * 60, 60 * 60))

    # !!! PREDEFINED METHOD NAME... DO NOT CHANGE !!!
    async def on_ready(self):
        """The on_ready method will wait until the bot is hosted and then run."""
        print(f"logged in as {self.user}")

        print(f'Preparing to ping joe...')
        await self.ping_joe()

    # !!! PREDEFINED METHOD NAME... DO NOT CHANGE !!!
    async def on_message(self, message):
        """The on_message method will constantly scan any newly received messages. Very useful."""
        process = re.compile('[^a-zA-Z]')
        # process the message into sanitised format
        processed = str(message.content).lower().strip()
        # print the message and author to the console (useful to debug, obviously)
        print(f"Message from {message.author}: {message.content}")

        # --- start of checks ---
        if message.author == bot.user:
            return


        # MOVE THIS TO FUNCTIONS.py thnx - etchris
        # --- My Method for getting fortnite balls ---
        if ("fortnite" in processed and "balls" in processed) or ("fortniteballs" in processed):
            await message.channel.send(
                "Fortnite balls\nhttps://www.youtube.com/watch?v=Kodx9em0mXE&ab_channel=Sergeantstinky-Topic")

        if "bruhshell" in process.sub('', str(message.content)).lower():
            await message.channel.send("! ! ! BRUH SHELL IS A CRYPTO-MINING SPYWARE ! ! !")

        if "lol" in message.content.strip().lower():
            await message.reply("https://i.imgflip.com/7b8363.gif")

        # Let's assume we want to parse bot commands - we could simply do something like what I illustrate below.
        if message.content[0] == COMMAND_PREFIX:
            # Here we could go ahead and search through a command base - I will just code one in as an example.
            if not message.content in commands:
                await message.channel.send(f"Hey, <@{message.author.id}>, that command was not found, bud!")
            else:
                await message.channel.send(commands[message.content])

        # Lastly, for fun, let's assume that the user pings the bot (bad idea)
        if str(message.content).__contains__("<@1075154837609656350>"):
            await message.channel.send(f"Hey, <@{message.author.id}>, don't ping me, bud!")


# -------------------- CREATE THE BOT OBJECT AND BOOT UP ------------------------

bot = FortniteBot(intents=intents)
bot.run(Config.TOKEN)
