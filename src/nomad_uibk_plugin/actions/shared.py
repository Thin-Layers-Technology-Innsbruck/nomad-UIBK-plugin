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
    pixel_size: list[float]
    model_name: str
    # model_classification_name: str
    csv_path: list[str]
    h5_path: list[str]
    output_path: list[str]
    overwrite_existing_results: bool
    mask_input_images: bool
    save_resulting_image: bool


@dataclass
class ActivityInferenceInput:
    image_file_name: str
    model_name: str
    pixel_size: float
    # model_classification_name: str
    csv_path: str
    h5_path: str
    output_path: str
    overwrite_existing_results: bool
    mask_input_images: bool
    save_resulting_image: bool


@dataclass
class WriteArchiveInput:
    pixel_size: float
    csv_path: str
    h5_path: str
    output_path: str
    upload_id: str
    user_id: str
    sample_id: str
    mask_input_images: bool


@dataclass
class ProcessNewFilesInput:
    upload_id: str
    user_id: str
    result_path: list[str]
