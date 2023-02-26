import os
import sys
import discord

try:
    import python_weather
except ModuleNotFoundError:
    print("installing python_weather w/o your permission :P")
    if sys.platform == 'win32': os.system("pip install python_weather")
    else: os.system("pip3 install python_weather")

class bruh:
    def __init__(self, tree, guild):

        self.commands = {
            'help': {'args': [""], "argconfigs": [0], "argc": 1},
            'view.users': {"args": [""], "argconfigs": [0], "argc": 1, "help-info": [""]},
            'life': {"args": [""], "argconfigs": [0], "argc": 1, "help-info": [""]},
            'wordle': {"args": [""], "argconfigs": [0], "argc": 1, "help-info": [""]},
            'bruh': {"args": ["", "-n"], "argconfigs": [0, 1], "argc": 2, "help-info": ["", "-n <number>"]},
            'weather': {"args": [""], "argconfigs": [0], "argc": 1, "help-info": [""]},
        }

        self.valid_commands = list(self.commands.keys())

        @tree.command(name="bruh", description="hello world, from BRUHSHELL 2.0", guild=discord.Object(id=guild))
        async def bruh(interaction, input_str: str = "help"):
            """
            command for all things bruh shell!
            """
            response = await base_process(input_str)
            final_response = ''
            if response:
                for finsished_job in response:
                    final_response += f"```{finsished_job}```\n"
                await interaction.response.send_message(final_response)
            else:
                await interaction.response.send_message("hmm, no completed jobs were returned . . .")

        async def base_process(command):
            finished_jobs = []
            command_line_in = command
            jobs = [job.strip() for job in command_line_in.split("&")]
            for job in jobs:
                command, arguement, arguement_values = tokenize_command(job)
                finished_jobs.append(await process_command(command, arguement, arguement_values))
            return finished_jobs

        def tokenize_command(command):
            tokens = [cmd.strip() for cmd in command.split(" ")]
            if len(tokens) == 0:
                return "skip", None, []
            elif len(tokens) == 1:
                return tokens[0], None, []
            elif len(tokens) == 2:
                return tokens[0], tokens[1], []
            elif len(tokens) == 3:
                return tokens[0], tokens[1], [tokens[2]]
            else:
                return tokens[0], tokens[1], tokens[2:]
        
        async def process_command(cmd, arg, argvs):
            print(argvs)
            if not cmd in self.valid_commands + ['']:
               return  f"{cmd} is not a valid command bro"
            elif cmd == 'help':
                response  = f'>> {cmd}\n'
                response += f'┏{"━"*33}┓\n'
                response += f'┃{"VALID COMMANDS":^33s}┃\n'
                response += f'┣{"━"*16}┳{"━"*16}┫\n'
                for i in range(0, len(self.valid_commands), 2):
                    response += f"┃ {self.valid_commands[i]:<15s}┃ {self.valid_commands[i+1]:<14s} ┃\n"
                response += f'┗{"━"*16}┻{"━"*16}┛\n'
                return response
            elif cmd == 'weather':
                return await process_weather(cmd, arg, argvs)
            else:
                return f">> {cmd} {arg if arg else ''}\nthat is not implemented yet . . ."
        
        async def process_weather(cmd, arg, argvs):
            if arg == None:
                return "**ERROR**: Usage 'weather <city>' | Usage 'weather -f <city>"
            elif arg == "-f":
                print(cmd, arg, argvs)
                if not argvs: 
                    cmd_string = cmd + ' ' + arg + ' ' + (' '.join(argvs) if argvs else '')
                    return f"**ERROR**: no city provided in '{cmd_string}'"
                city = ' '.join(argvs)
                response = await get_weather(city, arg)
                return response
            else:
                city = arg
                if argvs:
                    for word in argvs:
                        if word != None:
                            city += ' ' + word
                response = await get_weather(city)
                return response

        async def get_weather(city, flag=None):
            async with python_weather.Client(format="F") as client:
                response = await client.get(city)

                if not flag: return f"Current Temperature in {city} is **{response.current.temperature}°F** {response.current.type!r}"
                if flag == "-f":
                    forecast_response = ''
                    for i, forecast in enumerate(response.forecasts):
                        if i == 1:
                            break
                        date = f"Date: {forecast.date}"
                        sunrise = f"Sunrise: {forecast.astronomy.sun_rise}"
                        sunset = f"Sunset: {forecast.astronomy.sun_set}"
                        forecast_response += f"={date:^25s}{sunrise:^25s}{sunset:^24s}=\n"
                        t = "    Time           TEMP"
                        forecast_response += f"={(t + ' '*46):74s}=\n"
                        for hourly in forecast.hourly:
                            time_span = f" {str(hourly.time.hour).rjust(2, '0')}:{str(hourly.time.minute).ljust(2, '0')}   -->    {str(hourly.temperature).rjust(3, ' ')}"
                            info = f"{str(hourly.description).ljust(13, ' ')}  -->  {hourly.type!r} "
                            if hourly.description in ["Mist", "Partly cloudy"]:
                                forecast_response += f"=   {time_span:42s}{info:<28s}=\n"
                            else:
                                forecast_response += f"=   {time_span:42s}{info:<29s}=\n"
                            forecast_response += f"={' '*74}=\n"
                        forecast_response += "="*76 + "\n"
                    return forecast_response
