import os

import discord
from dotenv import load_dotenv
from fortnite_balls import FortniteBallsBot


def main():
    """
    Description: Invoke the bot and prepare the tokens.
    """
    load_dotenv()
    token = os.environ["BOT_token"]
    guild = os.environ["guild_ID"]
    cmds_path = os.environ["cmds_path"]
    debug = os.environ["debug"]

    intents = discord.Intents.default()
    intents.message_content = True
    intents.members = True
    intents.voice_states = True

    # handler = logging.FileHandler(filename='discord.log', encoding='utf-8', mode='w')

    bot = FortniteBallsBot(guild, cmds_path, debug=debug, intents=intents)
    bot.run(token)


if __name__ == "__main__":
    main()
