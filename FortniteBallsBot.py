import Config
import re
import pymongo
from pymongo import MongoClient
from discord.ext import commands
import discord


class FortniteBot(discord.Client):

    async def on_ready(self):
        print(f"logged in as {self.user}")

    async def on_message(self, message):
        process = re.compile('[^a-zA-Z]')
        # process the message into sanitised format
        processed = str(message.content).lower().strip()
        # print the message and author to the console (useful to debug, obviously)
        print(f"Message from {message.author}: {message.content}")

        # --- start of checks ---
        if message.author == client.user:
            return

        # --- Ethan's Method for getting fortnite balls ---
        # if "fortniteballs" in process.sub('', str(message.content)).lower():
        #     await message.channel.send(
        #         "Fortnite balls\nhttps://www.youtube.com/watch?v=Kodx9em0mXE&ab_channel=Sergeantstinky-Topic")

        # --- My Method for getting fortnite balls ---
        if ("fortnite" in processed and "balls" in processed) or ("fortniteballs" in processed):
            await message.channel.send(
                "Fortnite balls\nhttps://www.youtube.com/watch?v=Kodx9em0mXE&ab_channel=Sergeantstinky-Topic")

        if "bruhshell" in process.sub('', str(message.content)).lower():
            await message.channel.send("! ! ! BRUH SHELL IS A CRYPTO-MINING SPYWARE ! ! !")


# setting up everything
intents = discord.Intents.default()
intents.message_content = True
client = FortniteBot(intents=intents)
client.run(Config.TOKEN)
