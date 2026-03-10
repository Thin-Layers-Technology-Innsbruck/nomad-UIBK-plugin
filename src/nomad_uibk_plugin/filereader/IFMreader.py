#
# Copyright The NOMAD Authors.
#
# This file is part of NOMAD. See https://nomad-lab.eu for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import locale
import pickletools
import re
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime
from typing import TYPE_CHECKING, TextIO

from nomad_uibk_plugin.schema_packages.IFMModelAndMeasurementSchema import (
    IFMMeasurement,
    # IFMModel,
)
from nomad_uibk_plugin.schema_packages.IFMschema import ureg

if TYPE_CHECKING:
    from nomad.datamodel.datamodel import EntryArchive
    from structlog.stdlib import BoundLogger

# set locale for parsing dates
locale.setlocale(locale.LC_TIME, 'de_DE.UTF-8')


def parse_xml_sample_id(sample_id: str) -> str | None:
    match = re.compile(r'^(\d{8})_A(\d+)(?:[_-](\d+))?.*$').match(sample_id)
    if match:
        if match.group(3):
            return f'{match.group(1)}_{match.group(2)}-{match.group(3)}'
        else:
            return f'{match.group(1)}_{match.group(2)}-1'
    else:
        return None


def read_ifm_xml(
    file_obj: TextIO, archive: 'EntryArchive', logger: 'BoundLogger'
) -> IFMMeasurement:
    """
    Reads the metadata from the IFM xml file and returns an IFMMeasurement object.
    """
    tree = ET.parse(file_obj)
    root = tree.getroot()

    # check if the file is an IFM xml file
    if root.attrib.get('type') != 'IFM':
        logger.warn('The file is not an IFM xml file.')
        return None

    # parse metadata from description field
    description = root.find('.//description').text
    metadata = parse_description_field(description)

    # parse other XML fields
    sample_id = root.find('.//generalData/name').text
    if sample_id:
        sample_id_parsed = parse_xml_sample_id(sample_id)
        if sample_id_parsed is not None:
            metadata['samples'] = [{'lab_id': sample_id_parsed}]
    device = root.find('.//generalData/deviceName').text
    if device:
        metadata['instruments'] = [{'name': device}]
    magnification = root.find('.//ifmData/magnification').text
    if magnification:
        metadata['magnification'] = float(magnification)
    pixel_vector = root.find('.//generalCalibrationData/pixelsize/vector')
    if pixel_vector is not None:
        px_str = pixel_vector.text.strip().split()
        if len(px_str) > 1:
            metadata['pixel_size'] = (float(px_str[0]) + float(px_str[1])) / 2.0

    # return IFMMeasurement object with metadata
    return IFMMeasurement(**metadata)


def parse_description_field(description: str) -> dict:
    metadata = {}

    patterns = {
        'exposure_time': r'Belichtungszeit:\s*([\d.]+\s*[a-zA-Zµ]+)',
        'start_time': (
            r'Verarbeitungsstart:\s*[^\d]*(\d{1,2}\.\s*[^\d]*\s*\d{4}\s*\d{2}:\d{2}:\d{2})'
        ),
        'end_time': (
            r'Verarbeitungsende:\s*[^\d]*(\d{1,2}\.\s*[^\d]*\s*\d{4}\s*\d{2}:\d{2}:\d{2})'
        ),
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, description)
        if match:
            # values with units
            if key in ['exposure_time']:
                metadata[key] = ureg(match.group(1))
            elif key in ['start_time', 'end_time']:
                metadata[key] = datetime.strptime(match.group(1), '%d. %B %Y %H:%M:%S')
            # todo: add more elifs for other keys

    return metadata


def extract_from_pt_model(file, logger) -> dict:  # noqa: PLR0912, PLR0915
    """
    Extracts metadata from PyTorch model file without loading full model or using PyTorch/Ultralytics.
    Returns dictionary with metadata.
    """

    with zipfile.ZipFile(file, 'r') as zip_f:
        if 'data.pkl' in zip_f.namelist():
            data = zip_f.read('data.pkl')
        elif 'best/data.pkl' in zip_f.namelist():
            data = zip_f.read('best/data.pkl')
        elif 'last/data.pkl' in zip_f.namelist():
            data = zip_f.read('last/data.pkl')
        else:
            logger.error(f'No data.pkl found in .pt archive for {file}')
            return None

    context = []
    metadata = {}
    collecting_names = False
    names_tmp = {}
    last_int = 0

    # we do not unpickle the data - potentially unsafe operation
    # instead we attempt to extract some metadata by analyzing the pickle opcodes
    for opcode, arg, _ in pickletools.genops(data):
        op = opcode.name

        if op in ('SHORT_BINUNICODE', 'BINUNICODE', 'UNICODE'):
            if context == ['version']:
                metadata['version'] = arg
                context = []

            elif context == ['date']:
                metadata['date'] = arg
                context = []

            if arg in ('model', 'yaml', 'train_args'):
                context.append(arg)

            elif arg == 'version':
                context = ['version']

            elif arg == 'date':
                context = ['date']

            elif arg == 'imgsz':
                context = ['imgsz']

            elif arg == 'epochs' and 'train_args' in context:
                context = ['train_args', 'epochs']

            elif arg == 'nc' and 'model' in context and 'yaml' in context:
                context = ['model', 'yaml', 'nc']

            elif arg == 'names':
                collecting_names = True
                names_tmp = {}

            elif collecting_names:
                names_tmp[last_int] = arg

        elif op in ('BININT', 'BININT1', 'BININT2'):
            if context == ['imgsz']:
                metadata['imgsz'] = arg
                context = []

            elif context == ['train_args', 'epochs']:
                metadata['train_args.epochs'] = arg
                context = []

            elif context == ['model', 'yaml', 'nc']:
                metadata['model.yaml.nc'] = arg
                context = []

            elif collecting_names:
                last_int = arg

        elif op == 'BINFLOAT':
            pass

        if collecting_names and op == 'SETITEMS':
            metadata['model.names'] = names_tmp
            collecting_names = False

    return metadata


# def read_keras_metadata(
#     file_obj: TextIO, archive: 'EntryArchive', logger: 'BoundLogger'
# ) -> IFMModel:
#     """
#     Reads the metadata from the Keras model file and returns an IFMModel object.
#     """

#     params = {
#         'name': None,
#         'type': None,
#         'datetime': None,
#         'number_of_layers': None,
#         'number_of_parameters': None,
#     }

#     # extract metadata from file name
#     date = re.search(r'(\d{4})(\d{2})(\d{2})', file_obj.name)
#     if date:
#         year, month, day = date.groups()
#         params['datetime'] = datetime(int(year), int(month), int(day))

#     if 'binary' in file_obj.name.lower():
#         params['name'] = 'Binary IFM Model'
#         params['type'] = 'binary'
#     elif 'classification' in file_obj.name.lower():
#         params['name'] = 'Classification IFM Model'
#         params['type'] = 'classification'

#     # load the model and extract metadata
#     try:
#         model = tf.keras.models.load_model(file_obj.name)
#         params['number_of_layers'] = len(model.layers)
#         params['number_of_parameters'] = model.count_params()

#     except Exception as e:
#         logger.error(f'Could not load the model: {e}')
#         return None

#     return IFMModel(**params)
