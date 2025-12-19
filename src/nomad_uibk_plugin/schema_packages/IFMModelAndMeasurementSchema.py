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
from nomad_uibk_plugin.utils import update_sample_refs, get_reference

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
        # for key in archive.metadata.__dict__.keys():
        #     print(key, archive.metadata.__dict__[key])
        if self.name is None:
            self.name = (
                archive.metadata.mainfile.split('/')[-1].split('.')[0].replace('_', ' ')
            )

        archive.metadata.entry_name = self.name

        # Read metadata from file
        if self.metadata_file is not None:
            logger.info('Metadata file recognized. Parsing...')

            from nomad_uibk_plugin.filereader.IFMreader import read_ifm_xml

            with archive.m_context.raw_file(self.metadata_file) as file:
                measurement = read_ifm_xml(file, archive, logger)
                merge_sections(self, measurement, logger)

        # Update sample references
        for sample in self.samples:
            if sample.lab_id is not None and sample.reference is None:
                update_sample_refs(
                    sample=sample,  # pyright: ignore[reportArgumentType]
                    archive=archive,
                    logger=logger,
                    activity_type='ifm_measurement',
                    activity_name=self.name,
                )

            elif sample.reference is not None:
                if sample.lab_id is None:
                    sample.lab_id = sample.reference.lab_id
                if sample.name is None:
                    sample.name = sample.reference.name

        super().normalize(archive, logger)


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
