def seconds_to_hms(seconds: int):

    hours = int(seconds / 60 / 60)

    seconds = seconds - (hours * 60 * 60)

    minutes = int(seconds / 60)

    seconds = seconds - (minutes * 60)

    return (hours, minutes, seconds)