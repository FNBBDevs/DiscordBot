"""
DISCORD BOT

Authors: Nolan Gregory, Ethan Christensen, Klim F.
Version: 0.69
Intent: This multifaceted bot is to promote user activity within
        servers that it is hosted in.
"""

import Config
import discord
from _slash.slash_master import SlashMaster
from discord import app_commands
from FortniteBallsClient import FortniteClient

GUILD                   = Config.GUILD
TOKEN                   = Config.TOKEN
BOT_NAME                = "Fortnite Balls Bot"
intents                 = discord.Intents.default()
intents.message_content = True
client                  = FortniteClient(intents=intents)
tree                    = app_commands.CommandTree(client)

class FortniteBallsBot:
    def run(self):

        @client.event
        async def on_ready():
            """
            Can only have one event, this loads all slash commands that 
            are registered, and syncs them to the guild
            """

            # ping the delegator to tell slash master to load commands
            SlashMaster(tree, GUILD).load_commands()   
          
            # sync the commands with our guild (server)
            await tree.sync(guild=discord.Object(id=GUILD))
            print('[FNBB]: slash commands loaded')
        
        client.run(TOKEN)
