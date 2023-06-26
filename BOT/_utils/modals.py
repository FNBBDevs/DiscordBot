"""DEFINE YOUR MODALS HERE"""
import os
import time
import discord
import random
import json
import datetime
from discordwebhook import Discord
from discord import ui as UI
from discord.ui import Modal, Select, View
from _utils.weather import get_weather as Weather
from _utils.bruhpy import BruhPy
from _utils.lifegen import LifeGen
from _utils.nolang import Nolang

class UserInputModal(Modal):
    def __init__(self, prompt, short_or_long, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if short_or_long == 'short':
            self.add_item(UI.TextInput(label=prompt, style=discord.TextStyle.short))
        else:
            self.add_item(UI.TextInput(label=prompt, style=discord.TextStyle.long))


class WeatherModal(Modal):
    def __init__(self, typE, prompt, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_item(UI.TextInput(label=prompt, style=discord.TextStyle.short))
        self._typE = typE
        self._emoji_to_image = {
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

    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.defer()
        orignal_response = await interaction.original_response()
        try:
            weather = await Weather(self.children[0].value, self._typE)
            embed = discord.Embed(
                title=f"{weather.get('temp')} {weather.get('type')} {weather.get('desc')}", 
                description=f"weather for {weather.get('city')}",
                color=random.randint(0, 0xffffff),
                timestamp=datetime.datetime.now())
            if weather.get('type') in self._emoji_to_image:
                embed.set_image(url=self._emoji_to_image.get(weather.get('type')))
            else:
                embed.set_image(url=self._emoji_to_image.get(weather.get(None)))
            if weather.get('mode') == 'current':
                pass
            elif weather.get('mode') == 'both':
                embed.add_field(name='Sunrise', value=f"{weather.get('sunrise')}", inline=True)
                embed.add_field(name='Sunset', value=f"{weather.get('sunset')}", inline=True)
                for i, hourly in enumerate(weather.get('hourly')):
                    embed.add_field(name=f'{hourly[0]}', value=f'{hourly[1]}, {hourly[2]}', inline=False)
            else:
                pass
        except ValueError as ve:
            embed = discord.Embed(
                title=f"Unable to get weather for {self.children[0].value}", 
                description='Hey man! Not sure what happened but I guess I couldn\'t get the weather for that city. No worries though, I am sure you can google it!!!',
                color=random.randint(0, 0xffffff),
                timestamp=datetime.datetime.now())
            embed.set_image(url=self._emoji_to_image.get(None))

        embed.set_footer(text='verified.  ✅', icon_url='https://avatars.githubusercontent.com/u/132738989?s=400&u=36375e751dc38b698a858540b8fdd38f4d98396c&v=4')
        await orignal_response.edit(view=None, embed=embed)


class BruhPyModal(Modal):
    def __init__(self, show_code, prompt, view, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_item(UI.TextInput(label=prompt, style=discord.TextStyle.long))
        self._tags = {
            'ERROR':  'ansi',
            'NORMAL': '',
            'PY':     'py',
            'INFO':   'ansi', 
        }
        self._view = view
        self._show_code = show_code

    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.defer()
        original_response = await interaction.original_response()
        await original_response.edit(view=self._view)
        output = ''
        program = self.children[0].value.split(' ')
        for res in BruhPy(debug=False).run("-s" if self._show_code else program[0], program if self._show_code else program[1:], str(interaction.user)):
            output += f"```{self._tags[res[0]]}\n{res[1]}\n```\n"
        await original_response.edit(content=output, view=None)


class NolangModal(Modal):
    def __init__(self, show_code, prompt, view, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_item(UI.TextInput(label=prompt, style=discord.TextStyle.long))
        self._tags = {
            'ERROR':  'ansi',
            'NORMAL': '',
            'PY':     '',
            'INFO':   'ansi', 
        }
        self._view = view
        self._show_code = show_code

    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.defer()
        original_response = await interaction.original_response()
        await original_response.edit(view=self._view)
        output = ''
        program = self.children[0].value.split(' ')
        for res in Nolang(debug=False).run(arg="-s" if self._show_code else program[0], argvs=program if self._show_code else program[1:], user=str(interaction.user)):
            output += f"```{self._tags[res[0]]}\n{res[1]}\n```\n"
        await original_response.edit(content=output, view=None)


class GameOfLifeModal(Modal):
    marcus_says = Discord(url=os.environ['MARCUS'])
    def __init__(self, show_config, view, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.marcus_says = Discord(url=os.environ['MARCUS'])
        self._view = view
        self._show_config = show_config
        self.add_item(UI.TextInput(label="Enter a Grid Size:", style=discord.TextStyle.short))
        self.add_item(UI.TextInput(label="Enter a Refresh Speed(ms):", style=discord.TextStyle.short))
        self.add_item(UI.TextInput(label="Enter a Color Map:", style=discord.TextStyle.short))
        self.add_item(UI.TextInput(label="Enter an Interpolation:", style=discord.TextStyle.short))
        self.add_item(UI.TextInput(label="Render decay (0-no, 1-yes): ", style=discord.TextStyle.short))

    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.defer()
        original_response = await interaction.original_response()
        await original_response.edit(view=self._view)
        values = [child.value for child in self.children]
        try:
            life = LifeGen(values[4]=="1")
            life.gen_life_gif(int(values[0]), int(values[1]), values[2], values[3])
            with open('./BOT/_utils/_gif/tmp.gif', 'rb') as life_gif:
                gif = discord.File(life_gif)
                await original_response.add_files(gif)
                if self._show_config:
                    await original_response.edit(content=f"""```\nSize          : {values[0]}\nSpeed         : {values[1]}\nColormap      : {values[2]}\nInterpolation : {values[3]}\n```""",view=None)
                else:
                    await original_response.edit(view=None)
        except Exception as e:
            with open("./error.fnbbef", "a+") as f: f.write(f"{time.time()} -> {str(e)}\n")
            await original_response.edit(content="erm . . . what you requested is too large for a wee little boy like me [shaking, looks at ground nervously]. .  . uwu!", view=None)
            self.marcus_says.post(content="bro is not packing! 😭 🤣 🤣")
