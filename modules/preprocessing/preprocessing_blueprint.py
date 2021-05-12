
class PreprocessingAlgorithm:
    Name = "None"

    def __init__(self, model_path=""):
        return

    def preprocess_observations(self, decision_steps, terminal_steps):
        return decision_steps, terminal_steps

    def get_output_shapes(self, environment_configuration):
        return environment_configuration["ObservationShapes"]