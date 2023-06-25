import os
import shutil
import glob

def clean():
    songs = glob.glob('C:\\Users\\Nulzo\\Desktop\\github\\DiscordBot\\BOT\\song_cache\\*')
    for f in songs:
        os.remove(f)
