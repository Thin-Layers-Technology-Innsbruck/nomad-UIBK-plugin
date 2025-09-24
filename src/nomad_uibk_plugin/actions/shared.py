from dataclasses import dataclass


@dataclass
class MaskingInput:
    input_path: str
    overwrite_existing_results: bool
    mask_input_images: bool


@dataclass
class InferenceInput:
    upload_id: str
    user_id: str
    sample_id: list[str]
    image_file_name: list[str]
    model_binary_name: str
    model_classification_name: str
    csv_path: list[str]
    overwrite_existing_results: bool
    mask_input_images: bool


@dataclass
class ActivityInferenceInput:
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
    mask_input_images: bool


@dataclass
class ProcessNewFilesInput:
    upload_id: str
    user_id: str
    result_path: list[str]
