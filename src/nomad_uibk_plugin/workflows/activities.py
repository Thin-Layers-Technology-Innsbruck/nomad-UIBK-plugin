import json
import os

import pandas as pd
import plotly.graph_objs as go
from ifm_image_defect_detection.defectRecognition_toCSV import (
    defect_recognition,
)
from nomad.app.v1.routers.uploads import get_upload_with_read_access
from nomad.datamodel import User
from nomad.datamodel.metainfo.plot import PlotlyFigure
from nomad.datamodel.metainfo.workflow import Link
from nomad.orchestrator.utils import workflow_artifacts_dir
from temporalio import activity

from nomad_uibk_plugin.schema_packages.IFMschema import (
    DefectPrevalence,
    IFMAnalysisResult,
    IFMTwoStepAnalysisResult,
)
from nomad_uibk_plugin.workflows.shared import (
    InferenceInput,
    #     InferenceModelInput,
    InferenceResultsInput,
)


@activity.defn
async def run_inference(data: InferenceInput):
    if not os.path.exists(data.csv_path):
        activity.logger.info('Extracting defects...')
        defect_recognition(
            data.image_file_name, data.model_binary_name, data.model_classification_name
        )
    else:
        activity.logger.warning('Output file already exists')

@activity.defn
async def read_file(csv_path: str):
    if not os.path.exists(csv_path):
        raise FileExistsError('No csv file found.')
    else:
        defect_data = pd.read_csv(csv_path, skiprows=2)
        defect_columns = ['Whiskers', 'Chipping', 'Scratch', 'No Error']
        defect_data['type'] = defect_data[defect_columns].idxmax(axis=1)
        relative_share = defect_data['type'].value_counts(normalize=True)
        defect_mapping = {
            'Whiskers': 1,
            'Chipping': 2,
            'Scratch': 3,
            'No Error': 4,
        }
        defect_data['label'] = defect_data['type'].map(defect_mapping)
        heatmap = go.Heatmap(
            x=defect_data['x'],
            y=defect_data['y'],
            z=defect_data['label'],
            colorscale='Viridis',
            colorbar=dict(
                tickvals=[1, 2, 3, 4],
                ticktext=defect_columns,
                title='Defect Type',
            ),
        )

        figure = go.Figure(data=heatmap)
        figure.update_layout(
            title='Heatmap of Defect Distribution',
            xaxis_title='X Position',
            yaxis_title='Y Position',
            xaxis=dict(scaleanchor='y'),
            yaxis=dict(scaleanchor='x'),
            autosize=True,
        )

        figure_json = figure.to_plotly_json()
        figure_json['config'] = {'staticPlot': True}
        output_dict = {
            "csv_path": csv_path,
            "relative_share": relative_share,
            "figure_json": figure_json,
        }
        return output_dict


@activity.defn
async def write_to_archive(result_dict: dict):
    upload = get_upload_with_read_access(
        result_dict["upload_id"],
        User(user_id=result_dict["user_id"]),
        include_others=True,
    )
    analysis_entry = IFMAnalysisResult(file=result_dict["csv_path"])
    analysis_entry.defect_prevalence = DefectPrevalence(
        whiskers=result_dict["relative_share"].get('Whiskers', 0),
        chipping=result_dict["relative_share"].get('Chipping', 0),
        scratch=result_dict["relative_share"].get('Scratch', 0),
        no_error=result_dict["relative_share"].get('No Error', 0),
    )
    fname = os.path.join('inference_result.archive.json')
    with open(fname, 'w', encoding='utf-8') as f:
        json.dump({'data': analysis_entry.m_to_dict(with_root_def=True)}, f, indent=4)
    upload.process_upload(
        file_operations=[
            dict(op='ADD', path=fname, target_dir="", temporary=True)           # change to a proper target_dir later  # noqa: E501
        ],
        only_updated_files=True,
    )
    # self.outputs.append(analysis_entry)     # change self to proper object
    # archive.workflow2.outputs.append(
    #     Link(
    #         name='Extracted Features',
    #         section=analysis_entry,
    #     )
    # )
    # self.figures.append(
    #     PlotlyFigure(
    #         label='Defect Distribution Heatmap',
    #         index=0,
    #         figure=result_dict["figure_json"],
    #     )
    # )





# @activity.defn
# async def write_results(data: InferenceResultsInput):
#     upload = get_upload_with_read_access(
#         data.upload_id,
#         User(user_id=data.user_id),
#         include_others=True,
#     )
#     inference_result = IFMTwoStepAnalysisResult(            #change this into something reasonable
#         prompt=data.model_data.raw_input,
#         workflow_id=data.cif_dir,
#         inference_settings=InferenceSettings(
#             model=data.model_data.model_url.rsplit('/', 1)[-1].split('.tar.gz')[0],
#             num_samples=data.model_data.num_samples,
#             max_new_tokens=data.model_data.max_new_tokens,
#             temperature=data.model_data.temperature,
#             top_k=data.model_data.top_k,
#             seed=data.model_data.seed,
#             dtype=data.model_data.dtype,
#             compile=data.model_data.compile,
#         ),
#     )
#     fname = os.path.join('inference_result.archive.json')
#     with open(fname, 'w', encoding='utf-8') as f:
#         json.dump({'data': inference_result.m_to_dict(with_root_def=True)}, f, indent=4)
#     upload.process_upload(
#         file_operations=[
#             dict(op='ADD', path=fname, target_dir=data.cif_dir, temporary=True)
#         ],
#         only_updated_files=True,
#     )