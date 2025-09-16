from dataclasses import dataclass


@dataclass
class InferenceInput:
    upload_id: str
    user_id: str
    sample_id: str
    image_file_name: str
    model_binary_name: str
    model_classification_name: str
    csv_path: str
    overwrite_existing_results: bool


@dataclass
class WriteArchiveInput:
    csv_path: str
    upload_id: str
    user_id: str
    sample_id: str
