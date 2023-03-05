"""DEFINE YOUR MODALS HERE"""
import discord
from discord import ui as UI
from discord.ui import Modal
from _utils.weather import get_weather as Weather
from _utils.bruhpy import BruhPy

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
    def __init__(self, show_code, prompt, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_item(UI.TextInput(label=prompt, style=discord.TextStyle.long))
        self._tags = {
            'ERROR':  'diff',
            'NORMAL': '',
            'PY':     'py',
            'INFO':   'fix', 
        }
        self._show_code = show_code

    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.defer()
        orignal_response = await interaction.original_response()
        output = ''
        program = self.children[0].value.split(' ')
        for res in BruhPy(debug=False).run("-s" if self._show_code else program[0], program if self._show_code else program[1:]):
            output += f"```{self._tags[res[0]]}\n{res[1]}\n```\n"
        await orignal_response.edit(content=output, view=None)
