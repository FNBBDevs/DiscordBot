import discord
from discord.ui import Select, View
from _utils.modals import *
from _utils.bruhpy import BruhPy

class bruhv2:
    def __init__(self, tree, guild):

        self._weather_options = ['current', 'forecast']
        self._bruhpy_options  = ['show code', 'don\'t show code']
        self._valid_commands  = {
            'help': 'The **help** command provides information about each command, the arguments they take in, and the expected response. You provide a `command`, and the bot responds with information about that command.',
            'weather': 'The **weather** command allows you get the current temperature or forecast for a given city. You provide whether you want the `current` temperature or `forecast`, and then provvide a `city`. The bot reponds with the corresponding information about that city.',
            'bruhpy': 'The **bruhpy** command allows you get execute python code with the bot. You provide code the bot, and if it passes inspection, the code is executed and the bot displays the execution ouput.\nYour python code must follow a certain format where newlines in the program are replaced with `#` and tabs are still `\\t` If you want to use a newline within a string, use the standard `\\n`. An example program might look like this:\n```for i in range(10):#\\tprint("Hello, \\n world!")```'
        }



        @tree.command(name="bruhv2", description="bruh testing command", guild=discord.Object(id=guild))
        async def bruhv2(interaction: discord.Interaction):

            async def callback(interaction):

                await interaction.response.defer()

                selection = initial_select.values[0]
        
                if selection == "h":
                    view = self.get_help_options()
                elif selection == "w":
                    view = self.get_weather_options()
                elif selection == "b":
                    view = self.get_bruhpy_options()
                else:
                    view = None

                if view:
                    await interaction.edit_original_response(view=view)

            await interaction.response.defer()

            initial_select = Select(
                placeholder="What command would you like to execute?",
                options=[
                discord.SelectOption(
                    label="Help",
                    emoji="❔",
                    description="Learn about what the commands do!",
                    value="h"),
                discord.SelectOption(
                    label="Weather",
                    emoji="🌤️",
                    description="Get the weather or forecast for a City!",
                    value="w"),
                discord.SelectOption(
                    label="BruhPy",
                    emoji="🐍",
                    description="Execute Python Code!",
                    value="b"),
            ])
            initial_select.callback = callback
            view = View()
            view.add_item(initial_select)

            await interaction.followup.send(view=view)

    def get_weather_options(self):

        async def callback(interaction):
            typE = weather_select.values[0]
            modal = WeatherModal(
                typE=typE,
                prompt="City: ",
                short_or_long="short",
                title="Enter a City"
            )
            await interaction.response.send_modal(modal)

        options = []
        for option in self._weather_options:
            options.append(
                discord.SelectOption(label=option.capitalize(), value=option)
            )
        weather_select = Select(
            placeholder="Do you want the current Temperature or a Forecast?",
            options=options,
        )
        weather_select.callback = callback
        weather_view = View()
        weather_view.add_item(weather_select)
        
        return weather_view
    

    def get_help_options(self):

        async def callback(interaction):

            help_info = self._valid_commands[help_select.values[0]]
            await interaction.response.edit_message(content=help_info, view=None)

        options = []
        for option in self._valid_commands:
            options.append(
                discord.SelectOption(label=option.capitalize(), value=option)
            )
        
        help_select = Select(
            placeholder="What command?",
            options=options
        )

        help_select.callback = callback
        help_view = View()
        help_view.add_item(help_select)
        return help_view


    def get_bruhpy_options(self):

        async def callback(interaction):
            show_code = bruhpy_select.values[0] == "show code"
            modal = BruhPyModal(
                show_code=show_code,
                prompt="Enter your python code below",
                title="Enter your Code!"
            )
            await interaction.response.send_modal(modal)
        
        options = []
        for option in self._bruhpy_options:
            options.append(
                discord.SelectOption(label=option.capitalize(), value=option)
            )
        bruhpy_select = Select(
            placeholder="Do you want to display the code?",
            options=options
        )
        bruhpy_select.callback = callback
        bruhpy_view = View()
        bruhpy_view.add_item(bruhpy_select)

        return bruhpy_view
                