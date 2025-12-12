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

from typing import (
    TYPE_CHECKING,
)

from nomad.datamodel.data import EntryData
from nomad.metainfo import SchemaPackage, Section
from nomad_pvcomb.schema_packages.processes import JVMeasurement
from pint import UnitRegistry

from nomad_uibk_plugin.schema_packages import UIBKCategory
from nomad_uibk_plugin.utils import update_sample_refs

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

        # Update sample references
        for sample in self.samples:
            sample_file_name, sample_ref = update_sample_refs(
                sample=sample,  # pyright: ignore[reportArgumentType]
                archive=archive,
                logger=logger,
            )
            if (sample_file_name is not None) and (sample_ref is not None):
                sample.name = sample_file_name
                sample.reference = sample_ref

        super().normalize(archive, logger)
        archive.metadata.entry_name = self.name


m_package.__init_metainfo__()
