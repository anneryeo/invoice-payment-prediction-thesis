import json

class ParameterLoader:
    def __init__(self, json_file_location):
        self.json_file_location = json_file_location
        self.parameters = self._load_parameters()

    def _load_parameters(self):
        with open(self.json_file_location, "r") as f:
            return json.load(f)

    def get_parameters(self, model_name):
        """
        Returns list of parameter dicts for the given model.
        """
        return self.parameters.get(model_name, [])