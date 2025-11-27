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

from nomad.actions.utils import get_action_result, get_action_status, start_action
from nomad.app.v1.models.models import MetadataRequired
from nomad.datamodel.data import ArchiveSection, EntryData
from nomad.datamodel.metainfo.annotations import (
    ELNAnnotation,
    ELNComponentEnum,
    SectionProperties,
)
from nomad.datamodel.metainfo.basesections import (
    Entity,
    EntityReference,
)
from nomad.datamodel.metainfo.eln import ELNAnalysis
from nomad.datamodel.metainfo.plot import PlotSection
from nomad.datamodel.metainfo.workflow import Link
from nomad.metainfo import Quantity, SchemaPackage, Section, SubSection
from nomad.processing.data import Entry
from nomad.search import MetadataPagination, search
from pint import UnitRegistry

from nomad_uibk_plugin.actions.shared import InferenceInput
from nomad_uibk_plugin.schema_packages import UIBKCategory
from nomad_uibk_plugin.schema_packages.IFMModelAndMeasurementSchema import (
    IFMMeasurementReference,
    IFMModelReference,
)
from nomad_uibk_plugin.schema_packages.sample import UIBKSampleReference

if TYPE_CHECKING:
    from nomad.datamodel import EntryArchive
    from structlog.stdlib import BoundLogger

ureg = UnitRegistry()

m_package = SchemaPackage()


class DefectPrevalence(ArchiveSection):
    name = Quantity(
        type=str,
        description='Type of defect',
    )
    prevalence = Quantity(
        type=float,
        description='Prevalence of a given type of defect',
    )


class IFMTwoStepAnalysisResult(Entity, PlotSection, EntryData):
    m_def = Section(
        a_eln=ELNAnnotation(
            properties=SectionProperties(
                order=[
                    'name',
                    'file',
                    'description',
                    'datetime',
                    'lab_id',
                    'action_id',
                    'image_masked',
                    'figures',
                    'sample',
                    'defect_prevalence',
                ]
            )
        ),
    )

    file = Quantity(
        type=str,
        description='File containing the data.',
        a_eln=ELNAnnotation(component=ELNComponentEnum.FileEditQuantity),
    )

    sample = SubSection(
        section_def=UIBKSampleReference,
        description="""
        A list of all the samples measured during the measurement.
        """,
    )

    defect_prevalence = SubSection(
        section_def=DefectPrevalence,
        description='Prevalence of defects in the image.',
        repeats=True,
    )

    action_id = Quantity(
        type=str,
        description='ID of the inference action.',
    )

    image_masked = Quantity(
        type=bool,
        description="""
        True if image was masked automatically during analysis within nomad, False if 
        image was already masked before being uploaded.
        """,
        a_eln=ELNAnnotation(label='image masked during analysis'),
    )

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger'):
        super().normalize(archive, logger)
        self.name = self.name.split('.')[0].replace('_', ' ')  # type: ignore
        archive.metadata.entry_name = self.name


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

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger'):
        """Normalize the section to ensure it is ready for processing."""
        super().normalize(archive, logger)


class IFMTwoStepAnalysis(ELNAnalysis):
    """
    Automated image analysis entry.
    """

    m_def = Section(
        categories=[UIBKCategory],
        label='IFM Two Step Analysis',
        description='Form to run IFM inference actions from the ELN interface.',
        a_eln=ELNAnnotation(
            properties=SectionProperties(
                order=[
                    'name',
                    'datetime',
                    'overwrite_existing_results',
                    'mask_input_images',
                    'save_resulting_image',
                    'add_all_inputs',
                    'trigger_run_action',
                    'trigger_get_status',
                    'description',
                    'lab_id',
                    'location',
                    'tags',
                    'method',
                    'inputs',
                    'model',
                    'steps',
                    'analysis_identifiers',
                    'triggered_inference',
                    'outputs',
                ]
            )
        ),
    )

    inputs = SubSection(
        section_def=IFMMeasurementReference,
        description='Input data for the automated image analysis.',
        repeats=True,
    )
    outputs = SubSection(
        section_def=IFMTwoStepAnalysisResultReference,
        description='Output data from the automated image analysis.',
        repeats=True,
    )
    model = SubSection(
        section_def=IFMModelReference,
        description='Model for the automated image analysis.',
    )

    overwrite_existing_results = Quantity(
        type=bool,
        description=(
            'If checked, the existing inference results csv files will be overwritten.'
            'Otherwise, only images without corresponding outputs will be processed '
            'anew, processing for the other entries will be using existing files'
        ),
        default=False,
        a_eln=ELNAnnotation(component=ELNComponentEnum.BoolEditQuantity),
    )

    mask_input_images = Quantity(
        type=bool,
        description=(
            'If checked, attempt to mask the areas of the input image outside of '
            'the actual sample. The masked image will be saved in the same upload.'
        ),
        default=False,
        a_eln=ELNAnnotation(component=ELNComponentEnum.BoolEditQuantity),
    )

    save_resulting_image = Quantity(
        type=bool,
        description=(
            'If checked, saves the image with the defects found as an overlay in '
            'the same upload.'
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

    trigger_get_status = Quantity(
        type=bool,
        default=False,
        description='Retrieve the current status of the inference action.',
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.ActionEditQuantity,
            label='Get Action Status',
        ),
    )

    add_all_inputs = Quantity(
        type=bool,
        default=False,
        description="""
        Find and add all possible inputs (IFMMeasurement entries) in this upload. 
        Replaces all previously added manually inputs. Adds no more than 100 entries.
        """,
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.ActionEditQuantity,
            label='Add All IFMMeasurement Inputs',
        ),
    )

    triggered_inference = SubSection(
        section_def=InferenceStatus,
        description='A section for storing the status of the triggered inference '
        'action.',
    )

    def check_results(self, archive: 'EntryArchive', logger: 'BoundLogger'):
        if self.trigger_get_status and self.triggered_inference:
            try:
                status = get_action_status(
                    self.triggered_inference.action_id,
                    archive.metadata.authors[0].user_id,
                )  # noqa: E501
                if status:
                    self.triggered_inference.status = status.name
            except Exception as e:
                logger.error(f'Error getting action status: {e}. ')

            if self.triggered_inference.status == 'COMPLETED':
                result_refs = get_action_result(
                    self.triggered_inference.action_id,
                    archive.metadata.authors[0].user_id,
                )
                for i, output in enumerate(self.outputs):
                    output.reference = result_refs['refs'][i]

        self.trigger_get_status = False

    def find_source_path_from_ref(self, input_ref_subsection, extension):
        ref = input_ref_subsection.m_to_dict()['reference']
        upload_id = (re.search(r'uploads/(.*?)/archive', ref)).group(1)
        entry_id = (re.search(r'archive/(.*?)#/data', ref)).group(1)
        for entry in Entry.objects(upload_id=upload_id):  # type: ignore
            if entry.entry_id == entry_id:
                source_name = re.sub(r'\.archive\.json$', extension, entry.mainfile)
                source_path = (
                    f'.volumes/fs/staging/{upload_id[0:2]}/{upload_id}'
                    + f'/raw/{source_name}'
                )
                break
        return source_path

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger'):  # noqa: PLR0912
        super().normalize(archive, logger)
        self.method = 'IFM Two Step Analysis'

        # link IFMMeasurements automatically
        if self.add_all_inputs:
            self.add_all_inputs = False
            self.inputs = []
            query = {
                'entry_type': 'IFMMeasurement',
                'upload_id': archive.metadata.upload_id,
            }
            search_result = search(
                owner='all',
                query=query,
                pagination=MetadataPagination(page_size=100),
                user_id=archive.metadata.main_author.user_id,  # pyright: ignore
                required=MetadataRequired(include=['entry_id', 'entry_name']),
            )
            for entry in search_result.data:
                self.inputs.append(
                    IFMMeasurementReference(
                        name=entry['entry_name'],
                        reference=f'../uploads/{archive.metadata.upload_id}/archive/{entry["entry_id"]}#/data',
                    )
                )

        # archive workflow linking
        if self.model:
            archive.workflow2.inputs.append(  # type: ignore
                Link(name='Binary Model', section=self.model.reference)
            )

        # check if all necessary inputs are given
        if self.inputs and self.model:
            logger.info('Model found. Ready for IFM Two Step Analysis.')

            if self.trigger_run_action:
                # remove subsections corresponding to previous runs
                self.outputs = []
                binary_source_path = self.find_source_path_from_ref(
                    self.model, '.pt'
                )
                input_data = InferenceInput(
                    upload_id=archive.metadata.upload_id,
                    user_id=archive.metadata.authors[0].user_id,
                    sample_id=[],
                    image_file_name=[],
                    pixel_size=[],
                    model_name=binary_source_path,
                    csv_path=[],  # filled later inside the workflow
                    h5_path=[],
                    output_path=[],
                    overwrite_existing_results=self.overwrite_existing_results,
                    mask_input_images=self.mask_input_images,
                    save_resulting_image=self.save_resulting_image,
                )

                # create input data for the inference action
                for input in self.inputs:
                    # check that input has sample associated with it
                    if not input.reference:
                        logger.warning(
                            f'Input reference is missing for input {input.name}.'
                            + 'This input will be ignored.'
                        )
                        continue
                    if (
                        not input.reference.samples
                        or not input.reference.samples[0].lab_id
                    ):
                        logger.warning(
                            f'No sample found for input {input.name}.'
                            + 'This input will be ignored.'
                        )
                        continue

                    image_source_path = self.find_source_path_from_ref(input, '.bmp')

                    input_data.sample_id.append(input.reference.samples[0].lab_id)
                    input_data.image_file_name.append(image_source_path)
                    input_data.csv_path.append('')
                    input_data.h5_path.append('')
                    input_data.output_path.append('')
                    input_data.pixel_size.append(input.reference.pixel_size.magnitude)
                    self.outputs.append(
                        IFMTwoStepAnalysisResultReference(
                            name=input.name + '_inference_result',
                        )
                    )

                # 'unclick' buttons
                self.trigger_run_action = False
                self.overwrite_existing_results = False

                # Execute action that runs Georgs code to extract the defects
                # and creates results entries
                try:
                    action_id = start_action(
                        action_id='nomad_uibk_plugin.actions:ifm_inference',
                        data=input_data,
                    )
                    self.triggered_inference = InferenceStatus(
                        action_id=action_id, status='RUNNING'
                    )
                except Exception as e:
                    logger.error(f'Error running action: {e}')

        elif self.model:
            logger.warning(
                'No inputs to process have been found. IFM analysis aborted.'
            )
        else:
            logger.warning('No Models have been found. IFM analysis aborted.')

        self.check_results(archive=archive, logger=logger)


m_package.__init_metainfo__()
