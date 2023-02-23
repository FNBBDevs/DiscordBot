import Config
import discord
from discord import app_commands
import COMMANDS
from _commands import contains
from FortniteBallsClient import FortniteClient

GUILD                   = Config.GUILD
COMMAND_PREFIX          = '!'
BOT_NAME                = "Fortnite Balls Bot"
intents                 = discord.Intents.default()
intents.message_content = True
client                  = FortniteClient(intents=intents)
tree                    = app_commands.CommandTree(client)

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

        client.run(Config.TOKEN)