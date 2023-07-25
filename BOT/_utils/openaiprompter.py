import os

import openai


class OpenAIPrompter:
    def __init__(self):
        self.password = os.getenv("OPENPASS")
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def complete(self, prompt, password):
        print(f"[INFO] OpenAIPrompter: got this password from user: '{password}'")
        print(f"[INFO] OpenAIPrompter: comparing against: '{self.password}'")

        if os.getenv("OPENAI_API_KEY") is not None and password == self.password:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a cool dude named Marcus that is happy to talk to your friends.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )

            print(f"[INFO] OpenAIPrompter: from OpenAI: {completion}")

            try:
                return completion.choices[0].message
            except Exception as exception:
                print(f"[ERROR] OpenAIPrompter: {exception}")
        return None
