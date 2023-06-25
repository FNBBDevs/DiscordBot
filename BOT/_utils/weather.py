import python_weather


async def get_weather(city, typE="current"):
    async with python_weather.Client(format="F") as client:
        response = await client.get(city)
        if typE == "current":
            res = {
                'mode': 'current',
                'city': city,
                'temp': response.current.temperature,
                'type': f'{response.current.type!r}',
                'desc': response.current.description
            }
            return res
        elif typE == "forecast" or typE == "both":
            res = {
                'mode': 'both',
                'city': city,
                'temp': response.current.temperature,
                'type': f'{response.current.type!r}',
                'desc': response.current.description
            }
            for i, forecast in enumerate(response.forecasts):
                if i == 1:
                    break
                res['forecast_date'] = f"{forecast.date}"
                res['sunrise'] = f"{forecast.astronomy.sun_rise}"
                res['sunset'] = f"{forecast.astronomy.sun_set}"
                res['hourly'] = [(f"{str(hourly.time.hour).rjust(2, '0')}:{str(hourly.time.minute).ljust(2, '0')}", 
                                  f"{str(hourly.temperature).rjust(3, ' ')}°F",
                                  f"{str(hourly.description).ljust(14, ' ')}{hourly.type!r}") for hourly in forecast.hourly]
            return res

        else:
            return "**No Response**"

