from dataclasses import dataclass


@dataclass
class InferenceInput:
    upload_id: str
    user_id: str
    image_file_name: str
    model_binary_name: str
    model_classification_name: str
    csv_path: str

@dataclass
class InferenceResultsInput:
    upload_id: str
    user_id: str
    model_data: InferenceInput
