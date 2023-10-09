import os
import random
import re

from _utils.restrictions import BRUHPY_RESTRICTIONS
from discordwebhook import Discord


class Marcus:
    def __init__(self):
        self.check_vals = [
            r"""eval[(].*[)]""",
            r"""exec[(].*[)]""",
            r"""import\s[A-Za-z_]*""",
            r"""__import__[(].*[)]""",
            r"""globals[(][)]""",
            r"""getattr[(].*[)]""",
            r"""open[(].*[)]""",
            r"""while True:""",
            r"""__builtins__""",
            r"""__class__""",
            r"""__base__""",
            r"""__subclasses__""",
            r"""load_module[(].*[)]""",
        ]

        self._restrictions = BRUHPY_RESTRICTIONS
        self._hook = os.environ["MARCUS"]
        self._marcus_says = Discord(url=self._hook)

    def check_for_anti(self, anti, hits):
        flagged = None
        for hit in hits:
            if anti == hit:
                flagged = anti
                break

        if flagged:
            hits.remove(flagged)
            return True, hits
        else:
            return False, hits

    def log_it(self, string):
        with open("./marcusdebug.fnbbef", "a") as marcus:
            marcus.write(f"{string}\n")

    def erm__hey_marcus__can_you_check_this_code_out(self, program, user):
        hits = []
        hits_expanded = []
        for check in self.check_vals:
            if x := re.search(check, program):
                hits.append(x)

        for i, hit in enumerate(hits):
            if hit.groups():
                for g in hit.groups():
                    hits_expanded.append(g)
            else:
                hits_expanded.append(hit.group())

        antis = []
        tmp = program
        while x := re.search(r"""(?<=(["]\b))(?:(?=(\\?))\2.)*?(?=\1)""", tmp):
            antis.append(tmp[x.span()[0] : x.span()[1]])
            tmp = ("#" * x.span()[1]) + tmp[x.span()[1] :]

        tmp = program
        while x := re.search(r"""(?<=([']\b))(?:(?=(\\?))\2.)*?(?=\1)""", tmp):
            antis.append(tmp[x.span()[0] : x.span()[1]])
            tmp = ("#" * x.span()[1]) + tmp[x.span()[1] :]

        hit_count = len(hits)
        for anti in antis:
            anti_found, hits_expanded = self.check_for_anti(anti, hits_expanded)

            if anti_found:
                hit_count -= 1

        if hit_count > 0:
            if random.random() < 0.5:
                self._marcus_says.post(
                    content=(
                        "Woah!! Hey, are you sure the code you are trying to run isn't"
                        " breaking the rules defined by the creator of this bot? Or"
                        " worse, trying to run malicious code?"
                    )
                )
            else:
                self._marcus_says.post(content=("erm . . . what the flip dude!"))

        return not hit_count > 0
