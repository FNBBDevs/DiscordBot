import discord
from discord.ui import Select, View
from _utils.modals import *
from _utils.bruhpy import BruhPy

class bruh:
    def __init__(self, tree, guild):

        self._weather_options = [('current', '🌡️'), ('forecast', '⌚'), ('both', '☁️')]
        self._bruhpy_options  = [('show code', '✅'), ('don\'t show code', '❌')]
        self._life_options    = [('show config', '✅'), ('don\'t show config', '❌')]
        self._valid_commands  = {
            'help': 'The **help** command provides information about each command, the arguments they take in, and the expected response. You provide a `command`, and the bot responds with information about that command.',
            'weather': 'The **weather** command allows you get the current temperature or forecast for a given city. You provide whether you want the `current` temperature or `forecast`, and then provvide a `city`. The bot reponds with the corresponding information about that city.',
            'bruhpy': 'The **bruhpy** command allows you get execute python code with the bot. You provide code the bot, and if it passes inspection, the code is executed and the bot displays the execution ouput.\nYour python code must follow a certain format where newlines in the program are replaced with `#` and tabs are still `\\t` If you want to use a newline within a string, use the standard `\\n`. An example program might look like this:\n```for i in range(10):#\\tprint("Hello, \\n world!")```',
            'life': 'The **life** command allows you to input various attributes to generate a Conway\'s Game of Life GIF. These attributes are `size`, `refresh rate`, `color map`, and `interpolation`.'
        }
        self._command_information = {
            'help': ("Learn about what the commands do!","❔"),
            'weather': ("Get the weather or forecast for a City!", "🌤️"),
            'bruhpy': ("Execute Python Code!", "🐍"),
            'life': ("Generate a GOL GIF!", "🧬")
        }

        @tree.command(name="bruh", description="bruh testing command", guild=discord.Object(id=guild))
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
                elif selection == 'l':
                    view = self.get_life_options()
                else:
                    view = None

                if view:
                    await interaction.edit_original_response(view=view)

            await interaction.response.defer()

            options = []

            for option in self._command_information:
                options.append(discord.SelectOption(
                    label=option.capitalize(),
                    emoji=self._command_information[option][1],
                    description=self._command_information[option][0],
                    value=option[0]
                ))

            initial_select = Select(
                placeholder="What command would you like to execute?",
                options=options
            )

            initial_select.callback = callback
            view = View()
            view.add_item(initial_select)

            await interaction.followup.send(view=view, ephemeral=True)

    def get_help_options(self):

        async def callback(interaction):
            help_info = self._valid_commands[help_select.values[0]]
            await interaction.response.edit_message(content=help_info, view=None)

        options = []
        for option in self._command_information:
            options.append(
                discord.SelectOption(label=option.capitalize(), emoji=self._command_information[option][1], value=option)
            )
        
        help_select = Select(
            placeholder="What command to get info about?",
            options=options
        )

        help_select.callback = callback
        help_view = View()
        help_view.add_item(help_select)
        return help_view

    def get_weather_options(self):

        async def callback(interaction):
            typE = weather_select.values[0]
            modal = WeatherModal(
                typE=typE,
                prompt="City: ",
                title="Enter a City"
            )
            await interaction.response.send_modal(modal)

        options = []
        for option in self._weather_options:
            options.append(
                discord.SelectOption(label=option[0].capitalize(), emoji=option[1], value=option[0])
            )
        weather_select = Select(
            placeholder="Do you want the current Temperature or a Forecast?",
            options=options,
        )
        weather_select.callback = callback
        weather_view = View()
        weather_view.add_item(weather_select)
        
        return weather_view
    
    def get_bruhpy_options(self):

        async def callback(interaction):
            show_code = bruhpy_select.values[0] == "show code"
            bruhpy_select.placeholder = "executing program . . ."
            modal = BruhPyModal(
                show_code=show_code,
                prompt="Enter your python code below",
                view=bruhpy_view,
                title="Enter your Code!"
            )
            await interaction.response.send_modal(modal)
            bruhpy_select.disabled = True
            bruhpy_select.placeholder = "get program from user . . ."
            await interaction.edit_original_response(view=bruhpy_view)
        
        options = []
        for option in self._bruhpy_options:
            options.append(
                discord.SelectOption(label=option[0].capitalize(), emoji=option[1], value=option[0])
            )
        bruhpy_select = Select(
            placeholder="Do you want to display the code?",
            options=options
        )
        bruhpy_select.callback = callback
        bruhpy_view = View()
        bruhpy_view.add_item(bruhpy_select)

        return bruhpy_view

    def get_life_options(self):
        async def callback(interaction):
            show_config = life_select.values[0] == "show config"
            life_select.placeholder = "generating GOL gif . . ."
            modal = GameOfLifeModal(
                show_config=show_config,
                view=life_view,
                title="Set Game of Life Options"
            )
            await interaction.response.send_modal(modal)
            life_select.disabled = True
            life_select.placeholder = "getting values from user . . ."
            await interaction.edit_original_response(view=life_view)
        
        options = []
        for option in self._life_options:
            options.append(
                discord.SelectOption(label=option[0].capitalize(), emoji=option[1], value=option[0])
            )
        life_select = Select(
            placeholder="Do you want to display the config?",
            options=options
        )
        life_select.callback = callback
        life_view = View()
        life_view.add_item(life_select)

        return life_view
