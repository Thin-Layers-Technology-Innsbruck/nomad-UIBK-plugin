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

import os
from typing import (
    TYPE_CHECKING,
)

from nomad.actions.utils import get_action_status, start_action
from nomad.datamodel.data import ArchiveSection, EntryData
from nomad.datamodel.metainfo.annotations import (
    ELNAnnotation,
    ELNComponentEnum,
)
from nomad.datamodel.metainfo.basesections import (
    Entity,
    EntityReference,
)
from nomad.datamodel.metainfo.eln import ELNAnalysis
from nomad.datamodel.metainfo.plot import PlotSection
from nomad.datamodel.metainfo.workflow import Link
from nomad.metainfo import Datetime, MEnum, Quantity, SchemaPackage, Section, SubSection
from nomad.processing.data import Entry
from nomad_measurements.utils import (
    # create_archive,
    # get_entry_id_from_file_name,
    # get_reference,
    merge_sections,
)
from pint import UnitRegistry

from nomad_uibk_plugin.actions.shared import InferenceInput
from nomad_uibk_plugin.schema_packages import UIBKCategory
from nomad_uibk_plugin.schema_packages.IFMschema_extra import IFMMeasurement
from nomad_uibk_plugin.schema_packages.sample import UIBKSampleReference

if TYPE_CHECKING:
    from nomad.datamodel import EntryArchive
    from structlog.stdlib import BoundLogger

ureg = UnitRegistry()

m_package = SchemaPackage()


# class IFMMeasurement(ELNMeasurement):
#     """
#     IFM Measurement entry.
#     """

#     m_def = Section(
#         categories=[UIBKCategory],
#         label='IFM Measurement',
#         a_template=dict(
#             measurement_identifiers=dict(),
#         ),
#     )

#     image_file = Quantity(
#         type=str,
#         description='File containing the microscopy image.',
#         a_eln=ELNAnnotation(component=ELNComponentEnum.FileEditQuantity),
#     )

#     metadata_file = Quantity(
#         type=str,
#         description='File containing the measurement metadata.',
#         a_eln=ELNAnnotation(component=ELNComponentEnum.FileEditQuantity),
#     )

#     # Overwrite sample references with UIBKSampleReference
#     samples = SubSection(
#         section_def=UIBKSampleReference,
#         description="""
#         A list of all the samples measured during the measurement.
#         """,
#         repeats=True,
#     )
#     sample_id = Quantity(
#         type=str,
#         description='ID of the sample measured.',
#         a_eln=ELNAnnotation(component=ELNComponentEnum.StringEditQuantity),
#     )

#     # Metadata Quantities
#     start_time = Quantity(
#         type=Datetime,
#         description='The date and time when this process was started.',
#         a_eln=dict(label='start time'),  # component='DateTimeEditQuantity'
#     )

#     end_time = Quantity(
#         type=Datetime,
#         description='The date and time when this process was finished.',
#         a_eln=dict(label='end time'),
#     )

#     exposure_time = Quantity(
#         type=float,
#         description='Exposure time of the image.',
#         unit='second',
#         a_eln=ELNAnnotation(defaultDisplayUnit='µs'),
#     )

#     device = Quantity(
#         type=str,
#         description='Device used for the measurement.',
#         a_eln=dict(label='measurement device'),
#     )

#     magnification = Quantity(
#         type=float,
#         description='Magnification used for the measurement.',
#     )

#     def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger'):
#         """
#         Tasks in here:
#         - Read the metadata file and extract information from it.
#         - Update the sample references if lab_id is given.
#         """

#         self.method = 'IFM Measurement'

#         # Read metadata from file
#         if self.metadata_file is not None:
#             logger.info('Metadata file recognized. Parsing...')

#             from nomad_uibk_plugin.filereader.IFMreader import read_ifm_xml

#             with archive.m_context.raw_file(self.metadata_file) as file:
#                 measurement = read_ifm_xml(file, archive, logger)
#                 merge_sections(self, measurement, logger)

#         # Update sample references
#         if self.sample_id and not self.samples:
#             self.samples = [
#                 UIBKSampleReference(name=self.sample_id, lab_id=self.sample_id)
#             ]
#         elif self.samples and not self.sample_id:
#             self.sample_id = self.samples[0].lab_id

#         # Update measurement name
#         if self.samples:
#             self.name = f'IFM Measurement of {self.samples[0].name}'

#         super().normalize(archive, logger)


class IFMModel(Entity, EntryData):
    """
    Model for the automated image analysis.
    """

    m_def = Section(
        categories=[UIBKCategory],
        label='IFM Model',
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
        self.method = 'IFM Model'

        if self.file is not None:
            logger.info('Model file recognized. Parsing...')

            from nomad_uibk_plugin.filereader.IFMreader import read_keras_metadata

            with archive.m_context.raw_file(self.file, 'rb') as file:
                model = read_keras_metadata(file, archive, logger)
                merge_sections(self, model, logger)


class DefectPrevalence(ArchiveSection):
    whiskers = Quantity(
        type=float,
        description='Prevalence of whiskers.',
    )
    chipping = Quantity(
        type=float,
        description='Prevalence of chipping.',
    )
    scratch = Quantity(
        type=float,
        description='Prevalence of scratches.',
    )
    no_error = Quantity(
        type=float,
        description='Prevalence of no errors.',
    )


class IFMTwoStepAnalysisResult(Entity, PlotSection, EntryData):
    m_def = Section(
        categories=[UIBKCategory],
        label='IFM Two Step Analysis Result',
    )

    file = Quantity(
        type=str,
        description='File containing the data.',
        a_eln=ELNAnnotation(component=ELNComponentEnum.FileEditQuantity),
    )

    defect_prevalence = SubSection(
        section_def=DefectPrevalence,
        description='Prevalence of defects in the image.',
    )

    action_id = Quantity(
        type=str,
        description='ID of the inference action.',
    )

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger'):
        super().normalize(archive, logger)


class ImageReference(EntityReference):
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


class ModelReference(EntityReference):
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


class IFMTwoStepAnalysisResultReference(EntityReference):
    reference = Quantity(
        type=IFMTwoStepAnalysisResult,
        description='Reference to the IFM Two Step Analysis Result.',
        a_eln=ELNAnnotation(
            component='ReferenceEditQuantity',
            label='section reference',
        ),
    )

    action_id = Quantity(
        type=str,
        description='ID of the inference action.',
    )

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger'):
        super().normalize(archive, logger)

        # Update name
        if self.reference and self.name is None:
            self.name = self.reference.name


class InferenceStatus(ArchiveSection):
    """Section to fetch the status of an inference action."""

    action_id = Quantity(
        type=str,
        description='ID of the nference action.',
    )
    status = Quantity(
        type=str,
        description='Status of the inference action.',
    )
    trigger_get_status = Quantity(
        type=bool,
        default=False,
        description='Retrieve the current status of the inference action.',
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.ActionEditQuantity,
            label='Get Action Status',
        ),
    )

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger'):
        """Normalize the section to ensure it is ready for processing."""
        super().normalize(archive, logger)
        if not self.status or self.status == 'RUNNING' or self.trigger_get_status:
            try:
                status = get_action_status(self.action_id) # pyright: ignore[reportArgumentType]
                if status:
                    self.status = status.name
            except Exception as e:
                logger.error(f'Error getting action status: {e}. ')
            finally:
                self.trigger_get_status = False
            # if self.status == 'COMPLETED':
            #     reference = get_reference_from_mainfile(
            #         archive.metadata.upload_id,
            #         os.path.join(self.action_id, 'inference_result.archive.json'),
            #     )
            #     if not reference:
            #         logger.error(
            #             'Unable to set reference for the generated entry for '
            #             f'action {self.action_id}.'
            #         )
            #     else:
            #         self.generated_entry = reference


class IFMTwoStepAnalysis(ELNAnalysis):
    """
    Automated image analysis entry.
    """

    m_def = Section(
        categories=[UIBKCategory],
        label='IFM Two Step Analysis',
        description='Form to run IFM inference actions from the ELN interface.',
    )

    inputs = SubSection(
        section_def=ImageReference,
        description='Input data for the automated image analysis.',
        repeats=True,
    )
    outputs = SubSection(
        section_def=IFMTwoStepAnalysisResultReference,
        description='Output data from the automated image analysis.',
        repeats=True,
    )
    model_binary = SubSection(
        section_def=ModelReference,
        description='Model for the automated image analysis.',
    )
    model_classification = SubSection(
        section_def=ModelReference,
        description='Model for the automated image analysis.',
    )

    overwrite_existing_results = Quantity(
        type=bool,
        description=(
            'If checked, the existing inference results csv files will be overwritten.'
            'Otherwise, only images without corresponding outputs will be processed,'
            'processing for the other entries will be using existing files'
        ),
        default=False,
        a_eln=ELNAnnotation(component=ELNComponentEnum.BoolEditQuantity),
    )

    trigger_run_action = Quantity(
        type=bool,
        description='Starts an asynchronous action for running the inference.',
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.ActionEditQuantity,
            label='Run Inference Action',
        ),
    )

    trigger_get_statuses = Quantity(
        type=bool,
        default=False,
        description='Retrieve the current status of the inference actions.',
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.ActionEditQuantity,
            label='Get Actions Status',
        ),
    )

    triggered_inferences = SubSection(
        section_def=InferenceStatus,
        description='A section for storing the status of the triggered inference '
        'actions.',
        repeats=True,
    )

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger'):  # noqa: PLR0912
        super().normalize(archive, logger)
        self.method = 'IFM Two Step Analysis'

        # archive workflow linking
        if self.model_binary:
            archive.workflow2.inputs.append(  # type: ignore
                Link(name='Binary Model', section=self.model_binary.reference)
            )
        if self.model_classification:
            archive.workflow2.inputs.append(  # type: ignore
                Link(
                    name='Classification Model',
                    section=self.model_classification.reference,
                )
            )

        # check if all necessary inputs are given
        if self.inputs and self.model_binary and self.model_classification:
            logger.info('Two Models found. Ready for IFM Two Step Analysis.')

            if self.trigger_run_action:
                # remove subsections corresponding to previous runs
                self.triggered_inferences = []
                self.outputs = []
                for input in self.inputs:
                    # Execute action that runs Georgs code to extract the defects
                    # and creates results entries

                    image_file = archive.m_context.raw_file(input.reference.image_file)
                    print(f'^^^^^image_file: {image_file},\n{type(image_file)}')
                    image_file_name = image_file.name
                    model_binary = archive.m_context.raw_file(
                        self.model_binary.reference.file
                    )
                    model_binary_name = model_binary.name
                    model_classiciation = archive.m_context.raw_file(
                        self.model_classification.reference.file
                    )
                    model_classiciation_name = model_classiciation.name

                    # create paths and names for the csv file and archive file
                    path, filename_with_ext = os.path.split(image_file_name)
                    filename, ext = os.path.splitext(filename_with_ext)
                    csv_path = os.path.join(path, f'{filename}_prediction.csv')

                    # run action with analysis
                    try:
                        input_data = InferenceInput(
                            upload_id=archive.metadata.upload_id,
                            user_id=archive.metadata.authors[0].user_id,
                            triggering_entry_id=archive.metadata.entry_id,
                            image_file_name=image_file_name,
                            model_binary_name=model_binary_name,
                            model_classification_name=model_classiciation_name,
                            csv_path=csv_path,
                            overwrite_existing_results=self.overwrite_existing_results,
                        )
                        print(input_data)
                        action_id = start_action(
                            action_id='nomad_uibk_plugin.actions:ifm_inference',
                            data=input_data,
                        )
                        print(f'action has been started, id={action_id}')
                        print(f'entry id: {archive.metadata.entry_id}')

                        # create outputs (empty for now) and inference status for each
                        # input image
                        self.triggered_inferences.append(
                            InferenceStatus(action_id=action_id)
                        )  # type: ignore
                        self.outputs.append(
                            IFMTwoStepAnalysisResultReference(
                                name=input.name + '_inference_result',
                                action_id=action_id,
                            )
                        )  # type: ignore
                    except Exception as e:
                        logger.error(f'Error running action: {e}')
                    self.trigger_run_action = False
                    self.overwrite_existing_results = False

        elif self.model_binary and self.model_classification:
            logger.warning(
                'No inputs to process have been found. IFM analysis aborted.'
            )
        else:
            logger.warning('No Models have been found. IFM analysis aborted.')

        if self.trigger_get_statuses and self.triggered_inferences:
            for inference in self.triggered_inferences:
                try:
                    status = get_action_status(inference.action_id)
                    if status:
                        inference.status = status.name
                except Exception as e:
                    logger.error(f'Error getting action status: {e}. ')
                finally:
                    inference.trigger_get_status = False

        # TODO: get back to this when I have sample and image schema + parser
        # if self.triggered_inferences:
        #     for inference in self.triggered_inferences:
        #         if (
        #             self.trigger_get_statuses or inference.trigger_get_status
        #         ) and inference.status == 'COMPLETED':
        #             # search for the result to link to outputs
        #             result_entry_id = None
        #             for entry in Entry.objects(upload_id=archive.metadata.upload_id):
        #                 if entry.mainfile == mainfile:
        #                     result_entry_id = entry.entry_id
        #             if result_entry_id is None:
        #                 logger.warning('the output file was not found.')
        #             f'../uploads/{archive.metadata.upload_id}/archive/{result_entry_id}#/data'

        self.trigger_get_statuses = False


m_package.__init_metainfo__()
