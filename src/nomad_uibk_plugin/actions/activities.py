import json
import os
import re
import time

import pandas as pd
import plotly.graph_objs as go
from ifm_image_defect_detection.defectRecognition_toCSV import (
    defect_recognition,
)
from nomad.app.v1.routers.uploads import get_upload_with_read_access
from nomad.datamodel import User
from nomad.datamodel.metainfo.plot import PlotlyFigure
from nomad.infrastructure import setup_mongo
from nomad.processing.data import Entry
from temporalio import activity

from nomad_uibk_plugin.actions.shared import (
    CSVReadOutput,
    InferenceInput,
)
from nomad_uibk_plugin.schema_packages.IFMschema import (
    DefectPrevalence,
    IFMTwoStepAnalysisResult,
)


@activity.defn
async def run_ifm_inference(data: InferenceInput):
    if (not os.path.exists(data.csv_path)) or data.overwrite_existing_results:
        activity.logger.info('Extracting defects...')
        defect_recognition(
            data.image_file_name, data.model_binary_name, data.model_classification_name
        )
    else:
        activity.logger.warning('Output file already exists and not overwritten')


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
        defect_data_json = defect_data.to_json(orient='table')
        relative_share_json = relative_share.to_json(orient='table')

        output = CSVReadOutput(
            upload_id='',
            user_id='',
            triggering_entry_id='',
            csv_path=csv_path,
            defect_columns=defect_columns,
            defect_data_json=defect_data_json,
            relative_share_json=relative_share_json,
        )
        return output


@activity.defn
async def write_to_archive(result_from_csv: CSVReadOutput):
    setup_mongo()
    upload = get_upload_with_read_access(
        result_from_csv.upload_id,
        User(user_id=result_from_csv.user_id),
        include_others=True,
    )

    relative_share = pd.read_json(result_from_csv.relative_share_json, orient='table')
    relative_share_series = relative_share['proportion']
    defect_prevalence = DefectPrevalence(
        whiskers=relative_share_series.get('Whiskers', 0.0),
        chipping=relative_share_series.get('Chipping', 0.0),
        scratch=relative_share_series.get('Scratch', 0.0),
        no_error=relative_share_series.get('No Error', 0.0),
    )

    # create defects figure
    defect_data = pd.read_json(result_from_csv.defect_data_json, orient='table')
    heatmap = go.Heatmap(
        x=defect_data['x'],
        y=defect_data['y'],
        z=defect_data['label'],
        colorscale='Viridis',
        colorbar=dict(
            tickvals=[1, 2, 3, 4],
            ticktext=result_from_csv.defect_columns,
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

    # create a new archive entry with the results of the analysis
    result_name = (
        re.sub(r'_prediction\.csv$', '', result_from_csv.csv_path.split('/')[-1])
        + '_inference_result'
    )
    activity_info = activity.info()

    result_entry = IFMTwoStepAnalysisResult(
        name=result_name,
        file=result_from_csv.csv_path,
        defect_prevalence=defect_prevalence,
        action_id=activity_info.workflow_id,
        figures=[
            PlotlyFigure(
                label='Defect Distribution Heatmap',
                index=0,
                figure=figure_json,
            )
        ],
    )

    # add the new entry to the upload
    fname = os.path.join(result_name + '.archive.json')
    with open(fname, 'w', encoding='utf-8') as f:
        json.dump({'data': result_entry.m_to_dict(with_root_def=True)}, f, indent=4)
    upload.process_upload(
        file_operations=[
            dict(
                op='ADD', path=fname, target_dir='', temporary=True
            )  # change to a proper target_dir later  # noqa: E501
        ],
        only_updated_files=True,
    )

    # find entry_id for the resulting new entry
    entry_found = False
    num_max_attempts = 20
    for i in range(num_max_attempts):
        for entry in Entry.objects(upload_id=upload.upload_id):  # type: ignore
            if entry.mainfile == fname:
                result_entry_id = entry.entry_id
                entry_found = True
        if entry_found:
            break
        time.sleep(0.1)

    if entry_found:
        result_entry_reference = (
            f'../uploads/{upload.upload_id}/archive/{result_entry_id}#/data'
        )
    else:
        result_entry_reference = None

    return result_entry_reference
