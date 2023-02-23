
"""
DISCORD BOT

Authors: Nolan Gregory, Ethan Christensen, Klim F.
Version: 0.69
Intent: This multifaceted bot is to promote user activity within
        servers that it is hosted in.
"""



#import COMMANDS
import Config
import discord
from discord import app_commands
from FortniteBallsClient import FortniteClient

GUILD                   = Config.GUILD
TOKEN                   = Config.TOKEN
BOT_NAME                = "Fortnite Balls Bot"
intents                 = discord.Intents.default()
intents.message_content = True
client                  = FortniteClient(intents=intents)
tree                    = app_commands.CommandTree(client)
# COMMAND_PREFIX          = '!' <- deprecated?

class FortniteBallsBot:
    def run(self):
        """
        This command starts the bot and defines the slash (/) commands
        """
        @tree.command(name="deez", description="go ahead, give it a try buddy", guild=discord.Object(id=GUILD))
        async def first_command(interaction):
            await interaction.response.send_message("nutz [tips hat]")

        @client.event
        async def on_ready():
            await tree.sync(guild=discord.Object(id=GUILD))

        client.run(TOKEN)
