from fortnite_balls import FortniteBallsBot
from dotenv import load_dotenv

import os
import discord

def main():
    load_dotenv()
    TOKEN = os.environ['BOT_TOKEN']
    GUILD = os.environ['GUILD_ID']
    CMDS_PATH = os.environ['CMDS_PATH']
    DEBUG = os.environ['DEBUG']
    
    intents = discord.Intents.default()
    intents.message_content = True
    intents.members = True

    bot = FortniteBallsBot(GUILD, CMDS_PATH, debug=DEBUG, intents=intents)
    bot.run(TOKEN)

if __name__ == "__main__":
    main()