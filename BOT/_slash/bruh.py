import discord
import python_weather
from _utils.bruhpy import BruhPy
from _utils.lifegen import gen_life_gif, color_maps as CMAPS, interps as INTERPS

class bruh:
    def __init__(self, tree, guild):

        self.commands = {
            'help': {'args': [""], "argconfigs": [0], "argc": 1},
            'view.users': {"args": [""], "argconfigs": [0], "argc": 1, "help-info": [""]},
            'life': {"args": [""], "argconfigs": [0], "argc": 1, "help-info": [""]},
            'wordle': {"args": [""], "argconfigs": [0], "argc": 1, "help-info": [""]},
            'bruhpy': {"args": ["", "-n"], "argconfigs": [0, 1], "argc": 2, "help-info": [""]},
            'weather': {"args": [""], "argconfigs": [0], "argc": 1, "help-info": [""]},
        }
        self._tags = {
            'ERROR':  'diff',
            'NORMAL': '',
            'PY':     'py',
            'INFO':   'fix', 
        }
        self._tag = "\n . . . (truncated) . . .\n```"
        self._tag_length = len(self._tag)+1
        self.valid_commands = list(self.commands.keys())

        @tree.command(name="bruh", description="hello world, from BRUHSHELL 2.0", guild=discord.Object(id=guild))
        async def bruh(interaction, input_str: str = "help"):
            await interaction.response.defer()
            """
            command for all things bruh shell!
            
            - take the inputted string and parse it out into seperate jobs.
            - each job is ran and a response is returned.
            - responses have the form (<type>, <contents>)
            - and one may look like ("str", "hello world").
            - all str responses are merged into a larger response.
            - optionally, a single file can be sent denoted by "file"
            """
            final_response = f'```>> {input_str}\n```'
            opt_file = None
            async for finished_job in base_process(input_str):
                response_type = finished_job[0]
                response_contents = finished_job[1]

                if response_type == 'file': 
                    opt_file = response_contents
                else:
                    final_response += f"{response_contents}\n"
            if len(final_response)>2000:
                final_response = final_response[:2000-self._tag_length] + self._tag
            if opt_file:
                try:await interaction.followup.send(final_response, file=opt_file)
                except Exception as exception:await interaction.followup.send(f"```diff\n-{str(exception)}\n```")
            else:
                try:await interaction.followup.send(final_response)
                except Exception as exception:await interaction.followup.send(f"```diff\n-{str(exception)}\n```")

        async def base_process(command):
            command_line_in = command
            jobs = [job.strip() for job in command_line_in.split("&")]
            for job in jobs:
                command, arguement, arguement_values = tokenize_command(job)
                yield await process_command(command, arguement, arguement_values)

        def tokenize_command(command):
            tokens = [cmd.strip() for cmd in command.split(" ")]
            if len(tokens) == 0:return "skip", None, []
            elif len(tokens) == 1:return tokens[0], None, []
            elif len(tokens) == 2:return tokens[0], tokens[1], []
            elif len(tokens) == 3:return tokens[0], tokens[1], [tokens[2]]
            else:return tokens[0], tokens[1], tokens[2:]
        
        async def process_command(cmd, arg, argvs):
            try:
                if not cmd in self.valid_commands + ['']:return  ("str", f"{cmd} is not a valid command bro")
                elif cmd == 'help':
                    return await help()
                elif cmd == 'weather':
                    return await process_weather(cmd, arg, argvs)
                elif cmd == 'bruhpy':
                    return await bruhpy_execute(arg, argvs)
                elif cmd == 'life':
                    return await process_life(arg, argvs)
                else:
                    return ("str", f"that is not implemented yet . . .")
            except Exception as exception:
                return  ("str", f"ERROR: {exception}")
        
        async def help():
            response = f'┏{"━"*33}┓\n┃{"VALID COMMANDS":^33s}┃\n┣{"━"*16}┳{"━"*16}┫\n'\
            +''.join([f"┃ {self.valid_commands[i]:<15s}┃ {self.valid_commands[i+1]:<14s} ┃\n" for i in range(0, len(self.valid_commands), 2)])\
            +f'┗{"━"*16}┻{"━"*16}┛\n'
            return ("str", f"```\n{response}\n```")

        async def process_weather(cmd, arg, argvs):
            if arg == None:
                return ("str", "```diff\nERROR: Usage 'weather <city>' | Usage 'weather -f <city>'```")
            elif arg == "-f":
                if not argvs: 
                    cmd_string = cmd + ' ' + arg + ' ' + (' '.join(argvs) if argvs else '')
                    return ("str", f"```diff\nERROR: no city provided in '{cmd_string}'```")
                city = ' '.join(argvs)
                return await get_weather(city, arg)
            else:
                city = arg
                if argvs:
                    for word in argvs:
                        if word != None:
                            city += ' ' + word
                return await get_weather(city)

        async def get_weather(city, flag=None):
            async with python_weather.Client(format="F") as client:
                response = await client.get(city)
                if not flag: return ("str", f"```\nThe current temperature in {city} is {response.current.temperature}°F {response.current.type!r}\n```")
                if flag == "-f":
                    forecast_response = ''
                    for i, forecast in enumerate(response.forecasts):
                        if i == 1:
                            break
                        date = f"Date: {forecast.date}"
                        sunrise = f"Sunrise: {forecast.astronomy.sun_rise}"
                        sunset = f"Sunset: {forecast.astronomy.sun_set}"
                        forecast_response += f"{date:<25s}\n{sunrise:<25s}{sunset:<24s}\n"
                        for hourly in forecast.hourly:
                            time_span = f"{str(hourly.time.hour).rjust(2, '0')}:{str(hourly.time.minute).ljust(2, '0')}    {str(hourly.temperature).rjust(3, ' ')}°F"
                            info = f"{str(hourly.description).ljust(14, ' ')}{hourly.type!r} "
                            if hourly.description in ["Mist", "Partly cloudy"]:
                                forecast_response += f"{time_span:25s}{info:<28s}\n"
                            else:
                                forecast_response += f"{time_span:25s}{info:<29s}\n"
                    return ("str", f"```\n{forecast_response}\n```")

        async def bruhpy_execute(arg, argvs):
            response = ''
            master = BruhPy(debug=True)
            for res in master.run(arg, argvs):
                response += f"```{self._tags[res[0]]}\n{res[1]}\n```\n"
            return ("str", response)

        async def process_life(arg, argvs):
            if arg == '-h':
                if argvs[0] == 'color':
                    cm_reponse = ''
                    for i in range(0, len(CMAPS), 5):
                        if i < len(CMAPS) - 1:
                            cm_reponse += f"{CMAPS[i]:20s}"
                        if i + 1 < len(CMAPS) - 1:
                            cm_reponse += f"{CMAPS[i+1]:20s}"
                        if i + 2 < len(CMAPS) - 1:
                            cm_reponse += f"{CMAPS[i+2]:20s}"
                        if i + 3 < len(CMAPS) - 1:
                            cm_reponse += f"{CMAPS[i+3]:20s}"
                        if i + 4 < len(CMAPS) - 1:
                            cm_reponse += f"{CMAPS[i+4]:20s}"
                        cm_reponse += "\n"
                    return ("str", f"```\n{cm_reponse}\n```")
                elif argvs[0] == 'interpolations':
                    pass
                else:
                   pass
            argvs = [arg] + [val for val in argvs if val != '']
            if argvs[0] == '-s' and argvs[2] == '-r' and argvs[4] == '-cm' and argvs[6] == '-i':
                try:
                    gen_life_gif(int(argvs[1]), int(argvs[3]), argvs[5], argvs[7])
                    with open('./BOT/_utils/_gif/tmp.gif', 'rb') as life_gif:
                        gif = discord.File(life_gif)
                        return ("file", gif)
                except Exception as exception:
                    return ("str", f"```diff\n-[ERROR]: {str(exception)}\n```")
            else:
                return ("str", f"```diff\n-[ERROR]: invalid / missing arguments\n```\n```\n[USAGE]: life -s # -r # -cm <colormap> -i <interpolation>\n```")
