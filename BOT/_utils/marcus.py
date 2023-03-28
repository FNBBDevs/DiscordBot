import os
import re
import random
from discordwebhook import Discord
from _utils.restrictions import BRUHPY_RESTRICTIONS


class Marcus:
    def __init__(self):
        self.checks = [
            r""".*(eval[(].*[)]).*""",
            r""".*(exec[(].*[)]).*""",
            r""".*(import .*).*""",
            r""".*(__import__[(].*[)]).*""",
            r""".*(globals[(][)]).*""",
            r""".*(getattr[(].*[)]).*""",
            r""".*(open[(].*[)]).*""",
            r""".*(while True:).*""",
            r""".*(__builtins__).*""",
        ]
        self.anti_checks = [
            r"""(.*(".*eval.*[(].*[)].*").*)|(.*('.*eval.*[(].*[)].*').*)""",
            r"""(.*(".*exec.*[(].*[)].*").*)|(.*('.*exec.*[(].*[)].*').*)""",
            r"""(.*(".*import .*").*)|(.*('.*import .*').*)""",
            r"""(.*(".*__import__[(].*[)].*").*)|(.*('__import__[(].*[)].*').*)""",
            r"""(.*(".*globals[(][)].*").*)|(.*('.*globals[(][)].*').*)""",
            r"""(.*(".*getattr[(].*[)].*").*)|(.*('.*getattr[(].*[)].*').*)""",
            r"""(.*(".*open[(].*[)].*").*)|(.*('.*open[(].*[)].*').*)""",
            r"""(.*(".*while True:.*").*)|(.*('.*while True:.*').*)""",
            r"""(.*(".*__builtins__.*").*)|(.*('.*__builtins__.*').*)""",
        ]
        self.no_antis = [
            r"""(.*(f".*{.*eval.*[(].*[)].*}.*").*)|(.*(f'.*{.*eval.*[(].*[)].*}.*').*)""",
            r"""(.*(f".*{.*exec.*[(].*[)].*}.*").*)|(.*(f'.*{.*exec.*[(].*[)].*}.*').*)""",
            r"""(.*(f".*{.*import .*}.*").*)|(.*(f'.*{.*import .*}.*').*)""",
            r"""(.*(f".*{.*__import__[(].*[)].*}.*").*)|(.*(f'.*{.*__import__[(].*[)].*}.*').*)""",
            r"""(.*(f".*{.*globals[(][)].*}.*").*)|(.*(f'.*{.*globals[(][)].*}.*').*)""",
            r"""(.*(f".*{.*getattr[(].*[)].*}.*").*)|(.*(f'.*{.*getattr[(].*[)].*}.*').*)""",
            r"""(.*(f".*{.*open[(].*[)].*}.*").*)|(.*(f'.*{.*open[(].*[)].*}.*').*)""",
            r"""(.*(f".*{.*while True:.*}.*").*)|(.*(f'.*{.*while True:.*}.*').*)""",
            r"""(.*(f".*{.*__builtins__.*}.*").*)|(.*(f'.*{.*__builtins__.*}.*').*)""",
        ]
        self.no_antis_antis = [
            r"""(.*(f".*{.*'.*eval.*[(].*[)].*'.*}.*").*)|(.*(f'.*{.*".*eval.*[(].*[)].*".*}.*').*)""",
            r"""(.*(f".*{.*'.*exec.*[(].*[)].*'.*}.*").*)|(.*(f'.*{.*".*exec.*[(].*[)].*".*}.*').*)""",
            r"""(.*(f".*{.*'.*import .*'.*}.*").*)|(.*(f'.*{.*".*import .*".*}.*').*)""",
            r"""(.*(f".*{.*'.*__import__[(].*[)].*'.*}.*").*)|(.*(f'.*{.*".*__import__[(].*[)].*".*}.*').*)""",
            r"""(.*(f".*{.*'.*globals[(][)].*'.*}.*").*)|(.*(f'.*{.*".*globals[(][)].*".*}.*').*)""",
            r"""(.*(f".*{.*'.*getattr[(].*[)].*'.*}.*").*)|(.*(f'.*{.*".*getattr[(].*[)].*".*}.*').*)""",
            r"""(.*(f".*{.*'.*open[(].*[)].*'.*}.*").*)|(.*(f'.*{.*".*open[(].*[)].*".*}.*').*)""",
            r"""(.*(f".*{.*'.*while True:.*'.*}.*").*)|(.*(f'.*{.*".*while True:.*".*}.*').*)""",
            r"""(.*(f".*{.*'.*__builtins__.*'.*}.*").*)|(.*(f'.*{.*".*__builtins__.*".*}.*').*)""",
        ]
        self.for_real_no_antis = [
            r"""(.*(f".*{.*f'.*{.*(__builtins__).*}.*'.*}.*").*)|(.*(f'.*{.*f".*{.*(__builtins__).*}.*".*}.*').*)""",
            r"""(.*(f".*{.*f'.*{.*(__import__[(].*[)].*).*}.*'.*}.*").*)|(.*(f'.*{.*f".*{.*(__import__[(].*[)].*).*}.*".*}.*').*)""",
            r"""(.*(f".*{.*f'.*{.*(import .*).*}.*'.*}.*").*)|(.*(f'.*{.*f".*{.*(import).*}.*".*}.*').*)""",
            r"""(.*(f".*{.*f'.*{.*(while True:).*}.*'.*}.*").*)|(.*(f'.*{.*f".*{.*(while True:).*}.*".*}.*').*)""",
            r"""(.*(f".*{.*f'.*{.*(exec[(].*[)].*).*}.*'.*}.*").*)|(.*(f'.*{.*f".*{.*(exec[(].*[)].*).*}.*".*}.*').*)""",
            r"""(.*(f".*{.*f'.*{.*(eval[(].*[)].*).*}.*'.*}.*").*)|(.*(f'.*{.*f".*{.*(eval[(].*[)].*).*}.*".*}.*').*)""",
            r"""(.*(f".*{.*f'.*{.*(getattr[(].*[)].*).*}.*'.*}.*").*)|(.*(f'.*{.*f".*{.*(getattr[(].*[)].*).*}.*".*}.*').*)""",
            r"""(.*(f".*{.*f'.*{.*(open[(].*[)].*).*}.*'.*}.*").*)|(.*(f'.*{.*f".*{.*(open[(].*[)].*).*}.*".*}.*').*)""",
        ]
        self._restrictions = BRUHPY_RESTRICTIONS
        self._hook = os.environ['MARCUS']
        self._marcus_says = Discord(url=self._hook)

    def erm__hey_marcus__can_you_check_this_code_out(self, program):             
            hits = []
            flag = False
            lines = program.split("\n")
            for line in lines:
                if flag:break
                line = line.split(";")
                for hidden_line in line:
                    hidden_line = hidden_line.replace('"""', '"')
                    hidden_line = re.sub(" +", " ", hidden_line)
                    for check, anti_check in list(zip(self.checks, self.anti_checks)):
                        if (s1 := re.search(check, hidden_line)) and not (s2 := re.search(anti_check, hidden_line)):
                            hits.append((hidden_line, s1))
                            hits.append((hidden_line, s2))
                            flag = True
                            break
                    for restriction in self._restrictions:
                        check_1 = r"""(.*=.*"""+restriction+r""".*)"""
                        check_2 = r"""(.*[(]"""+restriction+r"""[)].*)"""
                        anti_check_1 = r"""(.*=\s*"""+"\""+restriction+r""".*")"""
                        anti_check_1_2 = r"""(.*=\s*'"""+restriction+r""".*')"""
                        anti_check_2 = r"""(.*[(]"""+"\""+restriction+r"""[)].*")"""
                        anti_check_2_2 = r"""(.*[(]'"""+restriction+r"""[)].*')"""
                        if re.search(check_1, hidden_line) or re.search(check_2, hidden_line):
                            if not (re.search(anti_check_1, hidden_line) or re.search(anti_check_1_2, hidden_line) or re.search(anti_check_2, hidden_line) or re.search(anti_check_2_2, hidden_line)):
                                hits.append((hidden_line, check_1))
                                hits.append((hidden_line, check_2))
                                flag = True
                                break
                    for check, anti_check in list(zip(self.no_antis, self.no_antis_antis)):
                        if re.search(check, hidden_line) and not re.search(anti_check, hidden_line):
                            hits.append((hidden_line, check))
                            hits.append((hidden_line, anti_check))
                            flag = True
                            break
                    for check in self.for_real_no_antis:
                        if check_res := re.search(check, hidden_line):
                            hits.append((hidden_line, check_res))
                            flag = True
                            break
            hits = [hit for hit in hits if hit[1]]
            if hits: 
                print(f"erm... Marcus here, you might want to look at this!\n{hits}")
                if random.random() < 0.5:
                    self._marcus_says.post(content=f"Woah!! Hey, are you sure the code you are trying to run isn't breaking the rules defined by the creator of this bot? Or worse, trying to run malicious code? This seems a little suspicious, `{hits[0][0]}`! Let's look over our code and try again!")
                else:
                    self._marcus_says.post(content=f"erm . . . what the flip dude! Thought you could get away with `{hits[0][0]}`?!")
            else: print(f"[tips hat]... Hey! Its Marcus, your code looks good my guy")
            return not flag
            return True if input("allow?: ").strip().lower() == "y" else False