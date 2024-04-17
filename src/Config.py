import json

from .OptimizedTorchConvNet import OptimizedTorchConvNet
from .TorchConvNet import TorchConvNet


class Config:
    def __init__(self, config_path="config.json"):
        self.settings = self.load_config(config_path)

    def load_config(self, path):
        with open(path, "r") as file:
            config = json.load(file)
        return config

    def get(self, key):
        return self.settings.get(key, None)

    def get_model(self):
        """Retrieve model class based on model name in the config."""
        model_name = self.get("model")
        if model_name == "TorchConvNet":
            return TorchConvNet()
        elif model_name == "OptimizedTorchConvNet":
            return OptimizedTorchConvNet()
        else:
            raise ValueError(f"Unknown model name: {model_name}")
