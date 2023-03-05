import python_weather


async def get_weather(city, typE="current"):
    async with python_weather.Client(format="F") as client:
        response = await client.get(city)
        if typE == "current":
            return f"```\nThe current temperature in {city} is {response.current.temperature}°F {response.current.type!r}\n```"
        elif typE == "forecast" or typE == "both":
            forecast_response = ''
            if typE == "both": forecast_response += f"The current temperature in {city} is {response.current.temperature}°F {response.current.type!r}\n\n"
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
            return f"```\n{forecast_response}\n```"            
        else:
            return "**No Response**"

