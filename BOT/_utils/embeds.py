import discord

COLORS = {
    "SUCCESS": 0x5CB85C,
    "WARNING": 0xFFE70A,
    "ERROR": 0xDC4C64,
    "INFO": 0x54B4D3,
    "BLACK": 0x332D2D,
    "WHITE": 0xFBFBFB
}

def on_success(title: str=None, description: str="", footer_text: str=None, footer_img: str=None, footer_usr: str=None):
    COLOR = COLORS["SUCCESS"]
    embed = discord.Embed(
          title=f"{title}", 
          description=f"{description}", 
          color= COLOR)
    embed.set_footer(text=f'{footer_text} {footer_usr}', icon_url= footer_img)
    return embed

def on_warning(title: str=None, description: str=None, footer_text: str=None, footer_img: str=None, footer_usr: str=None):
    COLOR = COLORS["WARNING"]
    embed = discord.Embed(
          title=f"{title}", 
          description=f"{description}", 
          color= COLOR)
    embed.set_footer(text=f'{footer_text} {footer_usr}', icon_url= footer_img)
    return embed

def on_error(title: str=None, description: str=None, footer_text: str=None, footer_img: str=None, footer_usr: str=None):
    COLOR = COLORS["ERROR"]
    embed = discord.Embed(
          title=f"{title}", 
          description=f"{description}", 
          color= COLOR)
    embed.set_footer(text=f'{footer_text} {footer_usr}', icon_url= footer_img)
    return embed

def on_info(title: str=None, description: str=None, footer_text: str=None, footer_img: str=None, footer_usr: str=None):
    COLOR = COLORS["INFO"]
    embed = discord.Embed(
          title=f"{title}", 
          description=f"{description}", 
          color= COLOR)
    embed.set_footer(text=f'{footer_text} {footer_usr}', icon_url= footer_img)
    return embed

def on_general(title: str=None, description: str=None, footer_text: str=None, footer_img: str=None, footer_usr: str=None):
    COLOR = COLORS["BLACK"]
    embed = discord.Embed(
          title=f"{title}", 
          description=f"{description}", 
          color= COLOR)
    embed.set_footer(text=f'{footer_text} {footer_usr}', icon_url= footer_img)
    return embed

def on_light(title: str=None, description: str=None, footer_text: str=None, footer_img: str=None, footer_usr: str=None):
    COLOR = COLORS["WHITE"]
    embed = discord.Embed(
          title=f"{title}", 
          description=f"{description}", 
          color= COLOR)
    embed.set_footer(text=f'{footer_text} {footer_usr}', icon_url= footer_img)
    return embed

def on_dark(title: str=None, description: str=None, footer_text: str=None, footer_img: str=None, footer_usr: str=None):
    COLOR = COLORS["BLACK"]
    embed = discord.Embed(
          title=f"{title}", 
          description=f"{description}", 
          color= COLOR)
    embed.set_footer(text=f'{footer_text} {footer_usr}', icon_url= footer_img)
    return embed