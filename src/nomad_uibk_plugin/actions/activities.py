import json
import os
import re
import time

import imageio
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from ifm_image_defect_detection.defectRecognition_toCSV import (
    defect_recognition,
)
from nomad.app.v1.routers.uploads import get_upload_with_read_access
from nomad.datamodel import User
from nomad.datamodel.metainfo.plot import PlotlyFigure
from nomad.processing.data import Entry
from skimage import color, feature, io, transform
from skimage.transform import hough_circle, hough_circle_peaks
from temporalio import activity

from nomad_uibk_plugin.actions.shared import (
    InferenceInput,
    MaskingInput,
    WriteArchiveInput,
)
from nomad_uibk_plugin.schema_packages.IFMschema import (
    DefectPrevalence,
    IFMTwoStepAnalysisResult,
)
from nomad_uibk_plugin.schema_packages.sample import UIBKSampleReference


@activity.defn
async def mask_image(masking_data: MaskingInput):
    input_path = masking_data.input_path
    if masking_data.mask_input_images:
        index = input_path.rfind('/IFM_')
        if index == -1:
            activity.logger.warning('Incorrect path for input image for masking.')
            return input_path
        else:
            output_path = input_path[: index + 1] + 'masked_' + input_path[index + 1 :]
        if (not os.path.exists(output_path)) or masking_data.overwrite_existing_results:
            resize_side = 500  # smaller side for rescaling
            img = io.imread(input_path)
            if img is None:
                raise FileExistsError(f'Could not read {input_path}')
            h, w = img.shape[:2]
            # Reduce resolution and convert to grayscale for faster image analysis
            scale = resize_side / min(h, w)
            small_w, small_h = int(w * scale), int(h * scale)
            small_img = transform.resize(
                img,
                (small_h, small_w),
                anti_aliasing=False,
                preserve_range=True,
            ).astype(np.uint8)  # type: ignore
            gray_small = color.rgb2gray(small_img)
            # Edge detection
            edges = feature.canny(gray_small, sigma=5)
            # Expected radius of the useful circle
            min_r = int(resize_side / 4)
            max_r = int(resize_side / 2)
            # Hough transform for circles
            radii = np.arange(min_r, max_r, 1)
            hough_res = hough_circle(edges, radii)
            # Find several most prominent circles and take the innermost
            _, cx, cy, radii_found = hough_circle_peaks(
                hough_res, radii, total_num_peaks=5
            )
            if len(radii_found) == 0:
                raise ValueError('No circles detected.')
            index = np.argmin(radii_found)
            x_s, y_s, r_s = cx[index], cy[index], radii_found[index]
            # Scale back to full resolution; +1 due to different indexing of the images
            scale_back = 1 / scale
            x = int(round((x_s + 1) * scale_back))
            y = int(round((y_s + 1) * scale_back))
            r = int(round(r_s * scale_back))
            # Create and apply the mask
            Y_grid, X_grid = np.ogrid[:h, :w]
            dist2 = (X_grid - x) ** 2 + (Y_grid - y) ** 2
            mask = dist2 <= r**2
            img[~mask] = 0
            # Save the result
            imageio.imwrite(output_path, img.astype(np.uint8))
        else:
            activity.logger.warning('Output file already exists and not overwritten.')
        return output_path
    else:
        return input_path


@activity.defn
async def generate_csv_path(image_path: str):
    path, filename_with_ext = os.path.split(image_path)
    filename, _ = os.path.splitext(filename_with_ext)
    csv_path = os.path.join(path, f'{filename}_prediction.csv')
    return csv_path


@activity.defn
async def run_ifm_inference(data: InferenceInput):
    if (not os.path.exists(data.csv_path)) or data.overwrite_existing_results:
        activity.logger.info('Extracting defects...')
        defect_recognition(
            data.image_file_name, data.model_binary_name, data.model_classification_name
        )
    else:
        activity.logger.warning('Output file already exists and not overwritten.')


@activity.defn
async def read_file_and_write_archive(writer_input: WriteArchiveInput):
    if not os.path.exists(writer_input.csv_path):
        raise FileExistsError('No csv file found.')
    else:
        defect_data = pd.read_csv(writer_input.csv_path, skiprows=2)
        defect_columns = defect_data.columns.to_list()
        defect_columns.remove('x')
        defect_columns.remove('y')
        defect_data['type'] = defect_data[defect_columns].idxmax(axis=1)
        relative_share = defect_data['type'].value_counts(normalize=True)
        defect_mapping = {key: idx for idx, key in enumerate(defect_columns, start=1)}
        defect_data['label'] = defect_data['type'].map(defect_mapping)

    upload = get_upload_with_read_access(
        writer_input.upload_id,
        User(user_id=writer_input.user_id),
        include_others=True,
    )

    defect_prevalence = []
    for key, value in relative_share.items():
        defect_prevalence.append(DefectPrevalence(name=key, prevalence=value))

    # create defects figure
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

    # create a new archive entry with the results of the analysis
    result_name = (
        re.sub(r'_prediction\.csv$', '', writer_input.csv_path) + '_inference_result'
    )
    activity_info = activity.info()

    result_entry = IFMTwoStepAnalysisResult(
        name=result_name.split('/')[-1],
        file=writer_input.csv_path,
        defect_prevalence=defect_prevalence,
        action_id=activity_info.workflow_id,
        sample=UIBKSampleReference(lab_id=writer_input.sample_id),
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
    fname = '/archive/'.join(fname.rsplit('/raw/', 1))
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
            if entry.mainfile == fname.split('/')[-1]:
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
