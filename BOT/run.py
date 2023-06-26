from fortnite_balls import FortniteBallsBot
from dotenv import load_dotenv
import logging
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
    intents.voice_states = True

    #handler = logging.FileHandler(filename='discord.log', encoding='utf-8', mode='w')

    bot = FortniteBallsBot(GUILD, CMDS_PATH, debug=DEBUG, intents=intents)
    bot.run(TOKEN)

if __name__ == "__main__":
    main()