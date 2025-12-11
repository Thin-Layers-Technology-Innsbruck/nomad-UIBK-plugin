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
from nomad.datamodel.metainfo.annotations import (
    ELNAnnotation,
    ELNComponentEnum,
    SectionProperties,
)
from nomad.datamodel.metainfo.basesections import (
    Entity,
    EntityReference,
    ReadableIdentifiers,
)
from nomad.datamodel.metainfo.eln import ELNMeasurement
from nomad.metainfo import Datetime, MEnum, Quantity, SchemaPackage, Section, SubSection
from nomad_measurements.utils import merge_sections
from pint import UnitRegistry

from nomad_uibk_plugin.schema_packages import UIBKCategory
from nomad_uibk_plugin.schema_packages.sample import UIBKSampleReference
from nomad_uibk_plugin.utils import update_sample_refs

if TYPE_CHECKING:
    from nomad.datamodel import EntryArchive
    from structlog.stdlib import BoundLogger

ureg = UnitRegistry()

m_package = SchemaPackage()


class IFMMeasurement(ELNMeasurement):
    """
    IFM Measurement entry.
    """

    m_def = Section(
        categories=[UIBKCategory],
        label='IFM Measurement',
        a_template=dict(
            measurement_identifiers=dict(),
        ),
        a_eln=ELNAnnotation(
            properties=SectionProperties(
                order=[
                    'name',
                    'image_file',
                    'metadata_file',
                    'start_time',
                    'end_time',
                    'description',
                    'location',
                    'lab_id',
                    'datetime',
                    'tags',
                    'method',
                    'exposure_time',
                    'magnification',
                    'instruments',
                    'samples',
                    'measurement_identifiers',
                    'steps',
                    'results',
                ],
            ),
        ),
    )

    image_file = Quantity(
        type=str,
        description='File containing the microscopy image.',
        a_eln=ELNAnnotation(component=ELNComponentEnum.FileEditQuantity),
    )

    metadata_file = Quantity(
        type=str,
        description='File containing the measurement metadata.',
        a_eln=ELNAnnotation(component=ELNComponentEnum.FileEditQuantity),
    )

    # Overwrite sample references with UIBKSampleReference
    samples = SubSection(
        section_def=UIBKSampleReference,
        description="""
        A list of all the samples measured during the measurement.
        """,
        repeats=True,
    )

    # Overwrite datetime with new label and description
    datetime = Quantity(
        type=Datetime,
        desription='The date and time when this entry was last processed.',
        a_eln=dict(label='entry processing time', component='DateTimeEditQuantity'),
    )

    # Metadata Quantities
    start_time = Quantity(
        type=Datetime,
        description='The date and time when this measurement was started.',
        a_eln=dict(label='start time', component='DateTimeEditQuantity'),
    )

    end_time = Quantity(
        type=Datetime,
        description='The date and time when this measurement was finished.',
        a_eln=dict(label='end time', component='DateTimeEditQuantity'),
    )

    exposure_time = Quantity(
        type=float,
        description='Exposure time of the image.',
        unit='second',
        a_eln=ELNAnnotation(defaultDisplayUnit='µs'),
    )

    magnification = Quantity(
        type=float,
        description='Magnification used for the measurement.',
    )

    pixel_size = Quantity(
        type=float,
        description='Size of a pixel of the image',
        unit='meter',
        a_eln=ELNAnnotation(defaultDisplayUnit='mm'),
    )

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger'):
        """
        Tasks in here:
        - Read the metadata file and extract information from it.
        - Update the sample references if lab_id is given.
        """

        self.method = 'IFM Measurement'

        # find corresponding data files
        bmp_name = re.sub(r'\.archive\.json$', '', archive.metadata.mainfile) + '.bmp'
        xml_name = (
            re.sub(r'texture\.archive\.json$', '', archive.metadata.mainfile)
            + 'info.xml'
        )

        from nomad.processing.data import Entry

        for entry in Entry.objects(upload_id=archive.metadata.upload_id):
            if entry.mainfile == bmp_name:
                self.image_file = bmp_name
            if entry.mainfile == xml_name:
                self.metadata_file = xml_name

        # Read metadata from file
        if self.metadata_file is not None:
            logger.info('Metadata file recognized. Parsing...')

            from nomad_uibk_plugin.filereader.IFMreader import read_ifm_xml

            with archive.m_context.raw_file(self.metadata_file) as file:
                measurement = read_ifm_xml(file, archive, logger)
                merge_sections(self, measurement, logger)

        # Update sample references
        for sample in self.samples:
            sample_file_name, sample_ref = update_sample_refs(
                sample=sample, archive=archive, logger=logger # pyright: ignore[reportArgumentType]
            )
            if (sample_file_name is not None) and (sample_ref is not None):
                sample.name = sample_file_name
                sample.reference = sample_ref

        super().normalize(archive, logger)
        archive.metadata.entry_name = self.name


class IFMModel(Entity, EntryData):
    """
    Model for the automated image analysis.
    """

    m_def = Section(
        categories=[UIBKCategory],
        label='IFM Model',
        a_template=dict(
            model_identifiers=dict(),
        ),
    )

    model_identifiers = SubSection(
        section_def=ReadableIdentifiers,
    )

    file = Quantity(
        type=str,
        description='File containing the data.',
        a_eln=ELNAnnotation(component=ELNComponentEnum.FileEditQuantity),
    )

    # Metadata Quantities
    type = Quantity(
        type=MEnum('binary', 'classification'), description='Type of the model.'
    )

    number_of_layers = Quantity(
        type=int,
        description='Number of layers in the model.',
    )

    number_of_parameters = Quantity(
        type=int,
        description='Number of parameters in the model.',
    )

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger'):
        """
        Read the model file and extract the metadata.
        """
        self.method = 'IFM Model'

        # find corresponding data files
        source_name = re.sub(r'\.archive\.json$', '', archive.metadata.mainfile) + '.pt'

        from nomad.processing.data import Entry

        for entry in Entry.objects(upload_id=archive.metadata.upload_id):
            if entry.mainfile == source_name:
                self.file = source_name

        super().normalize(archive, logger)
        archive.metadata.entry_name = self.name


class IFMMeasurementReference(EntityReference):
    reference = Quantity(
        type=IFMMeasurement,
        description='Reference to the IFM measurement.',
        a_eln=ELNAnnotation(
            component='ReferenceEditQuantity',
            label='section reference',
        ),
    )

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger'):
        super().normalize(archive, logger)

        # Update name
        if self.reference and self.name is None:
            self.name = self.reference.name


class IFMModelReference(EntityReference):
    reference = Quantity(
        type=IFMModel,
        description='Reference to the IFM model.',
        a_eln=ELNAnnotation(
            component='ReferenceEditQuantity',
            label='section reference',
        ),
    )

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger'):
        super().normalize(archive, logger)

        # Update name
        if self.reference and self.name is None:
            self.name = self.reference.name


m_package.__init_metainfo__()
