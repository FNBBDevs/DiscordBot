import discord
import asyncio
from discord.ext import commands
import Config
import pymongo
from pymongo import MongoClient
import re

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents = intents)
fortnitelist = "fortniteballs"

@client.event
async def on_ready():
    print("Online")

@client.event
async def on_message(message):
    process = re.compile('[^a-zA-Z]')
    print(message.content)
    if message.author == client.user:
        return
    if "fortniteballs" in process.sub('', str(message.content)).lower():
        await message.channel.send("Fortnite balls\nhttps://www.youtube.com/watch?v=Kodx9em0mXE&ab_channel=Sergeantstinky-Topic")
    if "bruhshell" in process.sub('', str(message.content)).lower():
        await message.channel.send("! ! ! BRUH SHELL IS A CRYPTO-MINING SPYWARE ! ! !")

async def setup():
    print("Setting up..")

async def main():
    await setup()
    await client.start(Config.TOKEN)

asyncio.run(main())

