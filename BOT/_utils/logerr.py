import datetime

class Logerr:
    def __init__(self):
        self.error_path = "../../error.fnbbef"
    
    def log(self, text):
        with open(self.error_path, 'a+') as error_file:
            error_file.write(f"[{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}] - {text}\n")
