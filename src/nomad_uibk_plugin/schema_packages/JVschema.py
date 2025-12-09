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

import re
from typing import (
    TYPE_CHECKING,
)

from nomad.datamodel.data import EntryData
from nomad.metainfo import SchemaPackage, Section
from nomad_pvcomb.schema_packages.activities import File
from nomad_pvcomb.schema_packages.processes import JVMeasurement
from pint import UnitRegistry

from nomad_uibk_plugin.schema_packages import UIBKCategory

if TYPE_CHECKING:
    from nomad.datamodel import EntryArchive
    from structlog.stdlib import BoundLogger

ureg = UnitRegistry()

m_package = SchemaPackage()


class UIBK_JVMeasurement(JVMeasurement, EntryData):
    m_def = Section(
        categories=[UIBKCategory],
        label='UIBK JV Measurement',
        description='UIBK JV Measurement entry.',
        a_template=dict(
            model_identifiers=dict(),
        ),
        a_eln=dict(
            hide=[
                'lab_id',
                'users',
                'author',
                'end_time',
                'location',
                'steps',
                'instruments',
            ],
            properties=dict(
                order=[
                    'name',
                    'data_file',
                    'samples',
                ],
            ),
        ),
        a_plot=[
            {
                'x': 'jv_curves/:/voltage',
                'y': 'jv_curves/:/current_density',
                'layout': {
                    'showlegend': True,
                    'yaxis': {'fixedrange': False},
                    'xaxis': {'fixedrange': False},
                },
            },
        ],
    )

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger'):
        """
        Read the JV measurement file and extract the data and metadata.
        """

        # find corresponding data files
        source_name = re.sub(r'\.archive\.json$', '.json', archive.metadata.mainfile)  # type: ignore

        from nomad.processing.data import Entry

        for entry in Entry.objects(upload_id=archive.metadata.upload_id):  # type: ignore
            if entry.mainfile == source_name:
                self.files = File(data_files=[source_name])

        super().normalize(archive, logger)
        # archive.metadata.entry_name = self.name
