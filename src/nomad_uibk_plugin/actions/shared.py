from dataclasses import dataclass


@dataclass
class InferenceInput:
    upload_id: str
    user_id: str
    triggering_entry_id: str
    sample_id: str
    image_file_name: str
    model_binary_name: str
    model_classification_name: str
    csv_path: str
    overwrite_existing_results: bool


# @dataclass
# class InferenceResultsInput:
#     upload_id: str
#     user_id: str
#     model_data: InferenceInput


@dataclass
class CSVReadOutput:
    upload_id: str
    user_id: str
    triggering_entry_id: str
    sample_id: str
    csv_path: str
    defect_data_json: str
    relative_share_json: str
    defect_columns: list[str]
