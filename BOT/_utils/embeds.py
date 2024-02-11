import datetime
import random

import discord


# pylint: disable=C0301

# Color mapping for embed. Neat-o colors if I do say so
colors = {
    "SUCCESS": 0x5CB85C,
    "WARNING": 0xFFE70A,
    "ERROR": 0xDC4C64,
    "INFO": 0x54B4D3,
    "BLACK": 0x332D2D,
    "WHITE": 0xFBFBFB,
    "PURPLE": 0x9D00FF,
}

def generic_colored_embed(
    title: str = None,
    description: str = "",
    footer_text: str = None,
    footer_img: str = None,
    footer_usr: str = None,
    color: any = "WHITE"       
):
    """
    Generic embed with colored sidebar
    :param title: title for the embed
    :param description: description for the embed
    :param footer_text: footer text for the embed
    :param footer_img: footer image for the embed
    :param footer_user: footer user for the embed
    :param color: color for the embed sidebar
    """
    
    if isinstance(color, str):
        color = colors[color]
    else:
        color = color
    
    if not footer_text:
        footer_text = ""
    if not footer_usr:
        footer_usr = ""

    embed = discord.Embed(title=f"{title}", description=f"{description}", color=color, timestamp=datetime.datetime.now())

    embed.set_footer(text=f"{footer_text} {footer_usr}", icon_url=footer_img)

    return embed

def weather(weather: dict, type: str = "current"):
    _emoji_to_image = {
        "☀️": "https://cdn.discordapp.com/attachments/1036736606465445898/1122596586854289538/sunny.png",
        "⛅️": "https://cdn.discordapp.com/attachments/1036736606465445898/1122596586485194792/partly_cloudy.png",
        "🌦": "https://cdn.discordapp.com/attachments/1036736606465445898/1122598864042610769/partly_cloudy_rain.jpg",
        "✨": "https://cdn.discordapp.com/attachments/1036736606465445898/1122599045102317720/starry.png",
        "☁️": "https://cdn.discordapp.com/attachments/1036736606465445898/1122599260903452693/cloudy.png",
        "❄️": "https://cdn.discordapp.com/attachments/1036736606465445898/1122616573786595358/heavy_rain_and_snow.png",
        "🌫️": "https://cdn.discordapp.com/attachments/1036736606465445898/1122601157425111190/fog.jpg",
        "🌧️": "https://cdn.discordapp.com/attachments/1036736606465445898/1122601267013885992/raining.jpg",
        "🌨️": "https://cdn.discordapp.com/attachments/1036736606465445898/1122616094096642148/light_snow.jpg",
        "🌨": "https://cdn.discordapp.com/attachments/1036736606465445898/1122616573786595358/heavy_rain_and_snow.png",
        None: "https://cdn.discordapp.com/attachments/1080959650389839872/1122615384714002433/co.gif",
    }

    embed = discord.Embed(
        title=f"{weather.get('temp')} {weather.get('type')} {weather.get('desc')}",
        description=f"weather for {weather.get('city')}",
        color=random.randint(0, 0xFFFFFF),
        timestamp=datetime.datetime.now(),
    )

    # if we have an image for this emoji, show it
    if weather.get("type") in _emoji_to_image:
        embed.set_image(url=_emoji_to_image.get(weather.get("type")))
    else:
        embed.set_image(url=_emoji_to_image.get(weather.get(None)))

    # if they selected both, we show sunrise, sunset, and forecast too
    if weather.get("mode") == "both":
        embed.add_field(name="Sunrise", value=f"{weather.get('sunrise')}", inline=True)
        embed.add_field(name="Sunset", value=f"{weather.get('sunset')}", inline=True)
        for i, hourly in enumerate(weather.get("hourly")):
            # hourly[0] -> hour
            # hourly[1] -> temp
            # hourly[2] -> description / emoji
            embed.add_field(
                name=f"{hourly[0]}", value=f"{hourly[1]}, {hourly[2]}", inline=False
            )

    embed.set_footer(
        text="verified.  ✅",
        icon_url="https://avatars.githubusercontent.com/u/132738989?s=400&u=36375e751dc38b698a858540b8fdd38f4d98396c&v=4",
    )

    return embed

def bruhby(output: list[str], user: str):
    """
    Embed to show the output of the bruhpy command
    :param output: list of output messages from the program run
    :param user: discord user who ran the command
    """

    embed = discord.Embed(timestamp=datetime.datetime.now())
    # python logo
    embed.set_author(
        name="Python Code Output",
        icon_url="https://cdn.discordapp.com/attachments/746957590457483345/1123416835119927356/1869px-Python-logo-notext.png",
    )

    if output[0] == "OUTPUT":
        # set the color to black
        embed.color = 0x000000
        # output result from running code
        embed.description = f"```\nFNBB:/bruh/{user.split('#')[0]} >> python your_code.py\n\n{output[1]}\n```"
        # marcus success footer
        embed.set_footer(
            text="verified by Marcus.  ✅",
            icon_url="https://cdn.discordapp.com/attachments/746957590457483345/1079971897682448454/image0.jpg",
        )
    else:
        # set color to red
        embed.color = 0xFF0000
        # error message from running code
        embed.description = f"```\nFNBB:/bruh/{user.split('#')[0]} >> python your_code.py\n\n{output[1]}\n```"
        # marcus failed footer
        embed.set_footer(
            text="verified by Marcus.  ❌",
            icon_url="https://cdn.discordapp.com/attachments/746957590457483345/1079971897682448454/image0.jpg",
        )

    return embed

def nolang(output: list[str], user: str):
    """
    Embed to show the output of the nolang command
    :param output: list of output messages from the program run
    :param user: discord user who ran the command
    """
    
    embed = discord.Embed(timestamp=datetime.datetime.now())
    # nolang logo
    embed.set_author(
        name="Nolang Code Output",
        icon_url="https://cdn.discordapp.com/attachments/746957590457483345/1123735080381206648/FNBB_1.png",
    )

    if output[0] == "OUTPUT":
        # set the color to black
        embed.color = 0x5271FF
        # output result from running code
        embed.description = f"```\nFNBB:/nolang/{user.split('#')[0]} >> nolang your_code.nl\n\n{output[1]}\n```"
        # nolang success footer
        embed.set_footer(
            text="verified.  ✅",
            icon_url="https://avatars.githubusercontent.com/u/132738989?s=400&u=36375e751dc38b698a858540b8fdd38f4d98396c&v=4",
        )
    else:
        # set color to red
        embed.color = 0xFF0000
        # error message from running code
        embed.description = f"```\nFNBB:/nolang/{user.split('#')[0]} >> nolang your_code.nl\n\n{output[1]}\n```"
        # nolang failed footer
        embed.set_footer(
            text="verified.  ❌",
            icon_url="https://avatars.githubusercontent.com/u/132738989?s=400&u=36375e751dc38b698a858540b8fdd38f4d98396c&v=4",
        )

    return embed
