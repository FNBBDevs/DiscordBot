import discord
import asyncio
from discord.ext import commands
import Config
import pymongo
from pymongo import MongoClient

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents = intents)

@client.event
async def on_ready():
    print("Online")

@client.event
async def on_message(message):
    if message.author == client.user:
        return
    await message.channel.send("Fortnite balls")

async def setup():
    print("Setting up..")

async def main():
    await setup()
    await client.start(Config.TOKEN)

asyncio.run(main())

