from dataclasses import dataclass


@dataclass
class InferenceInput:
    image_file_name: str
    model_binary_name: str
    model_classification_name: str