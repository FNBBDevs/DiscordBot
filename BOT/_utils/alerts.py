import bruhcolor

VALID_TYPES = {"ERROR": 196, "INFO": 220, "SUCCESS": 76, "GENERAL": 105}


class Alert:
    def __init__(self, alert_type, text):
        self.alert_type = alert_type
        self.text = self.set_message(text)

    def set_message(self, text):
        return str(bruhcolor.bruhcolored(text, color=VALID_TYPES[self.alert_type]))

    def __str__(self):
        return self.text


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
