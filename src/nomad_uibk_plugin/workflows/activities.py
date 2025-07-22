import os
import json

from nomad.orchestrator.utils import workflow_artifacts_dir
from temporalio import activity
from ifm_image_defect_detection.defectRecognition_toCSV import (
    defect_recognition,
)

from nomad.app.v1.routers.uploads import get_upload_with_read_access
from nomad.datamodel import User

from nomad_uibk_plugin.workflows.shared import (
    InferenceInput,
#     InferenceModelInput,
#     InferenceResultsInput,
)


@activity.defn
async def get_model(data):
        pass        #probably not needed

@activity.defn
async def construct_model_input(data):
        pass        #probably not needed

@activity.defn
async def run_inference(data: InferenceInput):
    path, filename_with_ext = os.path.split(data.image_file_name)
    filename, ext = os.path.splitext(filename_with_ext)
    csv_path = os.path.join(path, f'{filename}_prediction.csv')

    if not os.path.exists(csv_path):
        activity.logger.info('Extracting defects...')
        defect_recognition(
            data.image_file_name, data.model_binary_name, data.model_classification_name
        )


@activity.defn
async def write_results(data):
    upload = get_upload_with_read_access(
        data.upload_id,
        User(user_id=data.user_id),
        include_others=True,
    )
    inference_result = CrystaLLMInferenceResult(
        prompt=data.model_data.raw_input,
        workflow_id=data.cif_dir,
        inference_settings=InferenceSettings(
            model=data.model_data.model_url.rsplit('/', 1)[-1].split('.tar.gz')[0],
            num_samples=data.model_data.num_samples,
            max_new_tokens=data.model_data.max_new_tokens,
            temperature=data.model_data.temperature,
            top_k=data.model_data.top_k,
            seed=data.model_data.seed,
            dtype=data.model_data.dtype,
            compile=data.model_data.compile,
        ),
    )
    fname = os.path.join('inference_result.archive.json')
    with open(fname, 'w', encoding='utf-8') as f:
        json.dump({'data': inference_result.m_to_dict(with_root_def=True)}, f, indent=4)
    upload.process_upload(
        file_operations=[
            dict(op='ADD', path=fname, target_dir=data.cif_dir, temporary=True)
        ],
        only_updated_files=True,
    )