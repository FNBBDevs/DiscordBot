from bruhcolor import bruhcolored
from datetime import datetime


VALID_TYPES = {"ERROR": 196, "INFO": 220, "SUCCESS": 76, "GENERAL": 105}


class Alert:
    def __init__(self, alert_type, text):
        self.alert_type = alert_type
        self.text = self.set_message(text)

    def set_message(self, text):
        return bruhcolored(text, color=VALID_TYPES[self.alert_type])

    def __repr__(self):
        return self.text.colored


class ErrorAlert(Alert):
    def __init__(self, text):
        super(ErrorAlert, self).__init__("ERROR", text)


class InfoAlert(Alert):
    def __init__(self, text):
        super(InfoAlert, self).__init__("INFO", text)


class SuccessAlert(Alert):
    def __init__(self, text):
        super(SuccessAlert, self).__init__("SUCCESS", text)


class GeneralAlert(Alert):
    def __init__(self, text):
        super(GeneralAlert, self).__init__("GENERAL", text)


class DateTimeAlert:
    def __init__(self, text, dtia_alert_type, message_from):
        self.alert_type = "DTIA"
        self.text = self.set_message(text, dtia_alert_type, message_from)

    def set_message(self, text, dtia_alert_type, message_from):
        date_string = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        colored_part = (bruhcolored(date_string + " ", color=238)
                        + bruhcolored(f"{dtia_alert_type:<9s}", color=27, attrs=["bold"])
                        + bruhcolored(message_from + " ", color=5))
        return colored_part + text

