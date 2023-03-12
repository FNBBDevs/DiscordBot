"""DEFINE YOUR MODALS HERE"""
import os
import time
import discord
from discordwebhook import Discord
from discord import ui as UI
from discord.ui import Modal, Select, View
from _utils.weather import get_weather as Weather
from _utils.bruhpy import BruhPy
from _utils.lifegen import gen_life_gif as GenLifeGif

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

    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.defer()
        orignal_response = await interaction.original_response()
        weather = await Weather(self.children[0].value, self._typE)
        await orignal_response.edit(content=weather, view=None)


class BruhPyModal(Modal):
    def __init__(self, show_code, prompt, view, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_item(UI.TextInput(label=prompt, style=discord.TextStyle.long))
        self._tags = {
            'ERROR':  'diff',
            'NORMAL': '',
            'PY':     'py',
            'INFO':   'fix', 
        }
        self._view = view
        self._show_code = show_code

    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.defer()
        original_response = await interaction.original_response()
        await original_response.edit(view=self._view)
        output = ''
        program = self.children[0].value.split(' ')
        for res in BruhPy(debug=False).run("-s" if self._show_code else program[0], program if self._show_code else program[1:]):
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

    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.defer()
        original_response = await interaction.original_response()
        await original_response.edit(view=self._view)
        values = [child.value for child in self.children]
        try:
            GenLifeGif(int(values[0]), int(values[1]), values[2], values[3])
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
