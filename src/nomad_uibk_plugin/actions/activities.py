import json
import os
import re
import time

import h5py
import numpy as np
import plotly.graph_objs as go
from nomad.app.v1.routers.uploads import get_upload_with_read_access
from nomad.datamodel import User
from nomad.datamodel.metainfo.plot import PlotlyFigure
from nomad.processing.data import Entry
from PIL import Image
from temporalio import activity

from nomad_uibk_plugin.actions.shared import (
    ActivityInferenceInput,
    MaskingInput,
    ProcessNewFilesInput,
    WriteArchiveInput,
)
from nomad_uibk_plugin.schema_packages.IFMschema import (
    DefectPrevalence,
    IFMTwoStepAnalysisResult,
)
from nomad_uibk_plugin.schema_packages.sample import UIBKSampleReference

Image.MAX_IMAGE_PIXELS = 1000000000  # increases limit for image size

PATCH_SIZE = 640
LONG_SIDE_PLOT = 512  # approximate value due to int operations


@activity.defn
def mask_image(masking_data: MaskingInput):  # noqa: PLR0915
    import imageio
    from skimage import color, feature, io
    from skimage.transform import hough_circle, hough_circle_peaks

    input_path = masking_data.input_path
    if masking_data.mask_input_images:
        index = input_path.rfind('/')
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
            # Use PIL resizing to save on RAM
            scale = resize_side / min(h, w)
            small_w, small_h = int(w * scale), int(h * scale)
            pil_img = Image.fromarray(img)
            pil_small = pil_img.resize((small_w, small_h), Image.Resampling.LANCZOS)
            small_img = np.asarray(pil_small, dtype=np.uint8)
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
            # Create and apply the mask - row by row to save on RAM
            r2 = r * r
            for yi in range(h):
                dy = yi - y
                dy2 = dy * dy
                if dy2 > r2:
                    img[yi, :] = 0
                    continue

                dx = int((r2 - dy2) ** 0.5)
                x_start = max(0, x - dx)
                x_end = min(w, x + dx)

                if x_start > 0:
                    img[yi, :x_start] = 0

                if x_end < w:
                    img[yi, x_end:] = 0
            # Save the result
            imageio.imwrite(output_path, img.astype(np.uint8))
        else:
            activity.logger.info('Output file already exists and not overwritten.')
        return output_path
    else:
        return input_path


@activity.defn
def generate_paths(image_path: str):
    output_path, filename_with_ext = os.path.split(image_path)
    filename, _ = os.path.splitext(filename_with_ext)
    csv_path = os.path.join(output_path, f'{filename}_prediction.csv')
    h5_path = os.path.join(output_path, f'{filename}_prediction.h5')
    return output_path, csv_path, h5_path


@activity.defn
def run_ifm_inference(data: ActivityInferenceInput):
    from ifm_imagesegmentation.inference import run_image_segmentation

    if (not os.path.exists(data.csv_path)) or data.overwrite_existing_results:
        activity.logger.info('Extracting defects...')
        run_image_segmentation(
            image_path=data.image_file_name,
            model_path=data.model_name,
            output_path=data.output_path,
            patch_size=PATCH_SIZE,
            pixel_size=data.pixel_size,
            save_png=data.save_resulting_image,
            save_h5=True,
        )
        # delete the temporary masked file
        if data.mask_input_images:
            try:
                os.remove(data.image_file_name)
            except Exception as e:
                activity.logger.warning(
                    f'Could not remove temporary file {data.image_file_name}: {e}'
                )
    else:
        activity.logger.info('Output file already exists and not overwritten.')


@activity.defn
async def read_file_and_write_archive(writer_input: WriteArchiveInput):  # noqa: PLR0915
    if not os.path.exists(writer_input.csv_path):
        raise FileExistsError('No csv file found.')
    if not os.path.exists(writer_input.h5_path):
        raise FileExistsError('No h5 file found.')

    with h5py.File(writer_input.h5_path, 'r') as f:
        mask = f['mask'][()]  # type: ignore
        mask_types = f['mask_layers'][()]  # type: ignore

    # count defect prevalence from the masks
    mask_types = [msk.decode() for msk in mask_types]  # type: ignore
    defect_pixels = np.sum(mask, axis=(1, 2))  # type: ignore
    defect_types = mask_types
    no_def_pixel = 0
    for i, mask_type, def_pixel in zip(
        range(len(mask_types)), mask_types, defect_pixels
    ):
        if mask_type == 'Sample':
            total_pixel = def_pixel
            no_def_pixel += def_pixel
            no_def_index = i
        else:
            no_def_pixel -= def_pixel

    defect_types[no_def_index] = 'No Defect'
    relative_share = defect_pixels / total_pixel
    relative_share[no_def_index] = no_def_pixel / total_pixel

    defect_prevalence = []
    for key, value in zip(defect_types, relative_share):
        defect_prevalence.append(DefectPrevalence(name=key, prevalence=value))

    # Downscale masks for plotting
    layers, h, w = mask.shape  # type: ignore
    long_side = max(h, w)
    scale = max(long_side // LONG_SIDE_PLOT, 1)
    small_h = h // scale
    small_w = w // scale
    small_mask = np.zeros((layers, small_h, small_w), dtype=bool)

    for layer in range(layers):
        row_buffer = np.zeros((scale, w), dtype=bool)
        row_in_buffer = 0
        out_row = 0
        for row_index in range(h):
            row_buffer[row_in_buffer] = mask[layer, row_index]  # type: ignore
            row_in_buffer += 1

            # enough raws in buffer for one pool vertically
            if row_in_buffer == scale:
                # pool vertically, then horizontally
                v_pool = row_buffer.max(axis=0)
                h_and_v_pool = (
                    v_pool[: small_w * scale].reshape(small_w, scale).max(axis=1)
                )
                small_mask[layer, out_row] = h_and_v_pool
                out_row += 1
                row_in_buffer = 0

    # create figure with defects as json
    data_to_show = np.zeros((small_h, small_w), dtype=int)
    for i in range(layers):
        if i != no_def_index:
            data_to_show += (np.logical_not(data_to_show) & small_mask[i]).astype(
                int
            ) * (i + 1)
    data_to_show += (np.logical_not(data_to_show) & small_mask[no_def_index]).astype(
        int
    ) * (no_def_index + 1)
    h_mm = h * writer_input.pixel_size * 1000
    w_mm = w * writer_input.pixel_size * 1000
    # create defects figure
    heatmap = go.Heatmap(
        x0=0,
        dx=w_mm / small_w,
        y0=0,
        dy=h_mm / small_h,
        z=data_to_show[-1:0:-1],
        colorscale='Blackbody',
        colorbar=dict(
            tickvals=list(range(1, len(defect_types) + 1)),
            ticktext=defect_types,
            title='Defect Type',
        ),
    )

    figure = go.Figure(data=heatmap)
    figure.update_layout(
        title='Heatmap of Defect Distribution',
        xaxis_title='X Position [mm]',
        yaxis_title='Y Position [mm]',
        xaxis=dict(scaleanchor='y'),
        yaxis=dict(scaleanchor='x'),
        autosize=True,
    )
    figure_json = figure.to_plotly_json()

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
        image_masked=writer_input.mask_input_images,
        sample=UIBKSampleReference(lab_id=writer_input.sample_id),
        figures=[
            PlotlyFigure(
                label='Defect Distribution Heatmap',
                index=0,
                figure=figure_json,
            )
        ],
    )

    # add the new .archive.json entry to the upload
    fname = os.path.join(result_name + '.archive.json')
    with open(fname, 'w', encoding='utf-8') as f:
        json.dump({'data': result_entry.m_to_dict(with_root_def=True)}, f, indent=4)

    return fname


@activity.defn
async def process_new_files(data: ProcessNewFilesInput):
    file_operations = []
    mainfile_names = []

    for path in data.result_path:
        target_dir = path.split('/raw/')[-1]
        mainfile_names.append(target_dir)
        target_dir = '/'.join(target_dir.split('/')[:-1])
        file_operations.append(
            dict(op='ADD', path=path, target_dir=target_dir, temporary=False)
        )

    max_attempt_num = 100
    for i in range(max_attempt_num):
        upload = get_upload_with_read_access(
            data.upload_id,
            User(user_id=data.user_id),
            include_others=True,
        )

        if not upload.process_running:
            break
        else:
            # reload if upload is busy
            time.sleep(0.5)
            activity.logger.warning('Upload is currently being processed. Waiting...')

    handle = upload.process_upload(
        file_operations=file_operations,
        only_updated_files=True,
    )

    await handle.result()  # type: ignore

    result_entry_refs = []
    all_entries_this_upload = Entry.objects(upload_id=upload.upload_id)  # type: ignore

    for mainfile_name in mainfile_names:
        for entry in all_entries_this_upload:
            if entry.mainfile == mainfile_name:
                result_entry_refs.append(
                    f'../uploads/{upload.upload_id}/archive/{entry.entry_id}#/data'
                )

    return result_entry_refs
