import json


class Config:
    def __init__(self, config_path="config.json"):
        self.settings = self.load_config(config_path)

    def load_config(self, path):
        with open(path, "r") as file:
            config = json.load(file)
        return config

    def get(self, key):
        return self.settings.get(key, None)
