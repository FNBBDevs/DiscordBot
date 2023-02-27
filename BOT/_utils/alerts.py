import bruhcolor

VALID_TYPES = {'ERROR': 196, 'INFO': 220, 'SUCCESS': 76}

class Alert:
    def __init__(self, alert_type, text):
        self.alert_type = alert_type
        self.text = self.set_message(text)
    
    def set_message(self, text):
        return str(bruhcolor.bruhcolored(text, color=VALID_TYPES[self.alert_type]))
    
    def __str__(self):
        return self.text