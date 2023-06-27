import discord
import random
import datetime

# Color mapping for embed. Neat-o colors if I do say so
COLORS = {
    "SUCCESS": 0x5CB85C,
    "WARNING": 0xFFE70A,
    "ERROR": 0xDC4C64,
    "INFO": 0x54B4D3,
    "BLACK": 0x332D2D,
    "WHITE": 0xFBFBFB
}


# Embed that shows success 
def on_success(title: str=None, description: str="", footer_text: str=None, footer_img: str=None, footer_usr: str=None):
    COLOR = COLORS["SUCCESS"]
    embed = discord.Embed(
          title=f"{title}", 
          description=f"{description}", 
          color= COLOR)
    embed.set_footer(text=f'{footer_text} {footer_usr}', icon_url= footer_img)
    return embed


# Embed that shows warning 
def on_warning(title: str=None, description: str=None, footer_text: str=None, footer_img: str=None, footer_usr: str=None):
    COLOR = COLORS["WARNING"]
    embed = discord.Embed(
          title=f"{title}", 
          description=f"{description}", 
          color= COLOR)
    embed.set_footer(text=f'{footer_text} {footer_usr}', icon_url= footer_img)
    return embed


# Embed that shows error 
def on_error(title: str=None, description: str=None, footer_text: str=None, footer_img: str=None, footer_usr: str=None):
    COLOR = COLORS["ERROR"]
    embed = discord.Embed(
          title=f"{title}", 
          description=f"{description}", 
          color= COLOR)
    embed.set_footer(text=f'{footer_text} {footer_usr}', icon_url= footer_img)
    return embed


# Embed that shows info
def on_info(title: str=None, description: str=None, footer_text: str=None, footer_img: str=None, footer_usr: str=None):
    COLOR = COLORS["INFO"]
    embed = discord.Embed(
          title=f"{title}", 
          description=f"{description}", 
          color= COLOR)
    embed.set_footer(text=f'{footer_text} {footer_usr}', icon_url= footer_img)
    return embed


# Embed that shows general information
def on_general(title: str=None, description: str=None, footer_text: str=None, footer_img: str=None, footer_usr: str=None):
    COLOR = COLORS["BLACK"]
    embed = discord.Embed(
          title=f"{title}", 
          description=f"{description}", 
          color= COLOR)
    embed.set_footer(text=f'{footer_text} {footer_usr}', icon_url= footer_img)
    return embed


# Embed that shows light color
def on_light(title: str=None, description: str=None, footer_text: str=None, footer_img: str=None, footer_usr: str=None):
    COLOR = COLORS["WHITE"]
    embed = discord.Embed(
          title=f"{title}", 
          description=f"{description}", 
          color= COLOR)
    embed.set_footer(text=f'{footer_text} {footer_usr}', icon_url= footer_img)
    return embed


# Embed that shows dark color
def on_dark(title: str=None, description: str=None, footer_text: str=None, footer_img: str=None, footer_usr: str=None):
    COLOR = COLORS["BLACK"]
    embed = discord.Embed(
          title=f"{title}", 
          description=f"{description}", 
          color= COLOR)
    embed.set_footer(text=f'{footer_text} {footer_usr}', icon_url= footer_img)
    return embed

def weather(weather: dict, type: str='current'):
    _emoji_to_image = {
        '☀️':'https://cdn.discordapp.com/attachments/1036736606465445898/1122596586854289538/sunny.png',
        '⛅️':'https://cdn.discordapp.com/attachments/1036736606465445898/1122596586485194792/partly_cloudy.png',
        '🌦':'https://cdn.discordapp.com/attachments/1036736606465445898/1122598864042610769/partly_cloudy_rain.jpg',
        '✨':'https://cdn.discordapp.com/attachments/1036736606465445898/1122599045102317720/starry.png',
        '☁️':'https://cdn.discordapp.com/attachments/1036736606465445898/1122599260903452693/cloudy.png',
        '❄️':'https://cdn.discordapp.com/attachments/1036736606465445898/1122616573786595358/heavy_rain_and_snow.png',
        '🌫️':'https://cdn.discordapp.com/attachments/1036736606465445898/1122601157425111190/fog.jpg',
        '🌧️':'https://cdn.discordapp.com/attachments/1036736606465445898/1122601267013885992/raining.jpg',
        '🌨️':'https://cdn.discordapp.com/attachments/1036736606465445898/1122616094096642148/light_snow.jpg',
        '🌨':'https://cdn.discordapp.com/attachments/1036736606465445898/1122616573786595358/heavy_rain_and_snow.png',
        None: 'https://cdn.discordapp.com/attachments/1080959650389839872/1122615384714002433/co.gif'
    }

    embed = discord.Embed(
        title=f"{weather.get('temp')} {weather.get('type')} {weather.get('desc')}", 
        description=f"weather for {weather.get('city')}",
        color=random.randint(0, 0xffffff),
        timestamp=datetime.datetime.now())
    if weather.get('type') in _emoji_to_image:
        embed.set_image(url=_emoji_to_image.get(weather.get('type')))
    else:
        embed.set_image(url=_emoji_to_image.get(weather.get(None)))
    if weather.get('mode') == 'both':
        embed.add_field(name='Sunrise', value=f"{weather.get('sunrise')}", inline=True)
        embed.add_field(name='Sunset', value=f"{weather.get('sunset')}", inline=True)
        for i, hourly in enumerate(weather.get('hourly')):
            embed.add_field(name=f'{hourly[0]}', value=f'{hourly[1]}, {hourly[2]}', inline=False)
