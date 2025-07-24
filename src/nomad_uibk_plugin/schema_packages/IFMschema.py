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

import pandas as pd
import plotly.graph_objs as go
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
from nomad.datamodel.metainfo.eln import ELNAnalysis, ELNMeasurement
from nomad.datamodel.metainfo.plot import PlotlyFigure, PlotSection
from nomad.datamodel.metainfo.workflow import Link
from nomad.metainfo import Datetime, MEnum, Quantity, SchemaPackage, Section, SubSection
from nomad.orchestrator import utils as orchestrator_utils
from nomad.orchestrator.shared.constant import TaskQueue
from nomad_measurements.utils import (
    # create_archive,
    # get_entry_id_from_file_name,
    # get_reference,
    merge_sections,
)
from pint import UnitRegistry

from nomad_uibk_plugin.schema_packages import UIBKCategory
from nomad_uibk_plugin.schema_packages.sample import UIBKSampleReference
from nomad_uibk_plugin.workflows.shared import InferenceInput

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
    sample_id = Quantity(
        type=str,
        description='ID of the sample measured.',
        a_eln=ELNAnnotation(component=ELNComponentEnum.StringEditQuantity),
    )

    # Metadata Quantities
    start_time = Quantity(
        type=Datetime,
        description='The date and time when this process was started.',
        a_eln=dict(label='start time'),  # component='DateTimeEditQuantity'
    )

    end_time = Quantity(
        type=Datetime,
        description='The date and time when this process was finished.',
        a_eln=dict(label='end time'),
    )

    exposure_time = Quantity(
        type=float,
        description='Exposure time of the image.',
        unit='second',
        a_eln=ELNAnnotation(defaultDisplayUnit='µs'),
    )

    device = Quantity(
        type=str,
        description='Device used for the measurement.',
        a_eln=dict(label='measurement device'),
    )

    magnification = Quantity(
        type=float,
        description='Magnification used for the measurement.',
    )

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger'):
        """
        Tasks in here:
        - Read the metadata file and extract information from it.
        - Update the sample references if lab_id is given.
        """

        self.method = 'IFM Measurement'

        # Read metadata from file
        if self.metadata_file is not None:
            logger.info('Metadata file recognized. Parsing...')

            from nomad_uibk_plugin.filereader.IFMreader import read_ifm_xml

            with archive.m_context.raw_file(self.metadata_file) as file:
                measurement = read_ifm_xml(file, archive, logger)
                merge_sections(self, measurement, logger)

        # Update sample references
        if self.sample_id and not self.samples:
            self.samples = [
                UIBKSampleReference(name=self.sample_id, lab_id=self.sample_id)
            ]
        elif self.samples and not self.sample_id:
            self.sample_id = self.samples[0].lab_id

        # Update measurement name
        if self.samples:
            self.name = f'IFM Measurement of {self.samples[0].name}'

        super().normalize(archive, logger)


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


class IFMAnalysisResult(ArchiveSection):
    file = Quantity(
        type=str,
        description='File containing the data.',
        a_eln=ELNAnnotation(component=ELNComponentEnum.FileEditQuantity),
    )

    defect_prevalence = SubSection(
        section_def=DefectPrevalence,
        description='Prevalence of defects in the image.',
    )


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


class InferenceStatus(ArchiveSection):
    """Section to fetch the status of an inference workflow."""

    workflow_id = Quantity(
        type=str,
        description='ID of the `temporalio` workflow.',
    )
    status = Quantity(
        type=str,
        description='Status of the inference workflow.',
    )
    # generated_entry = Quantity(
    #     type=CrystaLLMInferenceResult,
    #     description='Reference to the generated entry after the workflow completes.',
    # )
    trigger_get_status = Quantity(
        type=bool,
        default=False,
        description='Retrieve the current status of the inference workflow.',
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.ActionEditQuantity,
            label='Get Workflow Status',
        ),
    )

    def normalize(self, archive, logger=None):
        """Normalize the section to ensure it is ready for processing."""
        super().normalize(archive, logger)
        # if not self.status or self.status == 'RUNNING' or self.trigger_get_status:
        #     try:
        #         status = orchestrator_utils.get_workflow_status(self.workflow_id)
        #         if status:
        #             self.status = status.name
        #     except Exception as e:
        #         logger.error(f'Error getting workflow status: {e}. ')
        #     finally:
        #         self.trigger_get_status = False
        #     if self.status == 'COMPLETED':
        #         reference = get_reference_from_mainfile(
        #             archive.metadata.upload_id,
        #             os.path.join(self.workflow_id, 'inference_result.archive.json'),
        #         )
        #         if not reference:
        #             logger.error(
        #                 'Unable to set reference for the generated entry for '
        #                 f'workflow {self.workflow_id}.'
        #             )
        #         else:
        #             self.generated_entry = reference


class InferenceSettings(ArchiveSection):
    image_file_name: str
    model_binary_name: str
    model_classification_name: str


class IFMTwoStepAnalysisResult(EntryData):
    m_def = Section(
        label='IFMLLM Inference Result',
        a_eln=ELNAnnotation(
            properties=SectionProperties(
                order=[
                    'workflow_id',
                    'status',
                    'trigger_workflow_status',
                    'inference_settings',
                ]
            ),
        ),
    )
    workflow_id = Quantity(
        type=str,
        description='ID of the `temporalio` workflow',
    )
    inference_settings = SubSection(
        section_def=InferenceSettings,
        description='Settings used for the LLM inference workflow'
    )


class IFMTwoStepAnalysis(ELNAnalysis, PlotSection):
    """
    Automated image analysis entry.
    """

    m_def = Section(
        categories=[UIBKCategory],
        label='IFM Two Step Analysis',
    )

    inputs = SubSection(
        section_def=ImageReference,
        description='Input data for the automated image analysis.',
        repeats=True,
    )
    outputs = SubSection(
        section_def=IFMAnalysisResult,
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

    # Execution Quantity
    perform_analysis = Quantity(
        type=bool,
        description=(
            'Check box to trigger the automated image analysis after assigning '
            'the measurement(s) and the two models.'
        ),
        default=False,
        a_eln=ELNAnnotation(component=ELNComponentEnum.BoolEditQuantity),
    )

    trigger_run_workflow = Quantity(
        type=bool,
        description='Starts an asynchronous workflow for running the inference.',
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.ActionEditQuantity,
            label='Run Inference Workflow',
        ),
    )

    triggered_inferences = SubSection(
        section_def=InferenceStatus,
        description='A section for storing the status of the triggered inference '
        'workflow.',
        repeats=True,
    )

    # def run_workflow(self, archive, logger=None):
    #     """
    #     Run the LLM workflow with the provided archive.
    #     """
    #     input_data = None           #change to actual input data here later on
    #     workflow_name = 'nomad_UIBK_plugin.workflows.InferenceWorkflow'
    #     workflow_id = orchestrator_utils.start_workflow(
    #         workflow_name=workflow_name,
    #         data=input_data,
    #         task_queue=TaskQueue.GPU
    #     )
    #     if not self.triggered_inferences:
    #         self.triggered_inferences = [InferenceStatus()]
    #     else:
    #         self.triggered_inferences.append(InferenceStatus()) # type: ignore
    #     self.triggered_inferences[-1].workflow_id = workflow_id


    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger'):
        super().normalize(archive, logger)
        self.method = 'IFM Two Step Analysis'

        # archive workflow linking
        if self.model_binary:
            archive.workflow2.inputs.append( # type: ignore
                Link(name='Binary Model', section=self.model_binary.reference)
            )
        if self.model_classification:
            archive.workflow2.inputs.append( # type: ignore
                Link(
                    name='Classification Model',
                    section=self.model_classification.reference,
                )
            )

        # check if all necessary inputs are given
        if self.inputs and self.model_binary and self.model_classification:
            logger.info('Two Models found. Ready for IFM Two Step Analysis.')

            self.outputs = []
            for input in self.inputs:
                # here we execute Georgs code to extract the defects
                
                image_file = archive.m_context.raw_file(input.reference.image_file)
                image_file_name = image_file.name
                model_binary = archive.m_context.raw_file(self.model_binary.reference.file)
                model_binary_name = model_binary.name
                model_classiciation = archive.m_context.raw_file(self.model_classification.reference.file)
                model_classiciation_name = model_classiciation.name

                # create paths and names for the csv file and archive file
                path, filename_with_ext = os.path.split(image_file_name)
                filename, ext = os.path.splitext(filename_with_ext)
                csv_path = os.path.join(path, f'{filename}_prediction.csv')

                #run workflow with analysis, if triggered
                if self.trigger_run_workflow:
                    try:
                        input_data = InferenceInput(
                            upload_id=archive.metadata.upload_id,
                            user_id=archive.metadata.authors[0].user_id,
                            image_file_name=image_file_name,
                            model_binary_name=model_binary_name,
                            model_classification_name=model_classiciation_name,
                            csv_path=csv_path,
                        )
                        workflow_name = 'nomad_UIBK_plugin.workflows.InferenceWorkflow'
                        workflow_id = orchestrator_utils.start_workflow(
                            workflow_name=workflow_name,
                            data=input_data,
                            task_queue=TaskQueue.GPU
                        )
                        if not self.triggered_inferences:
                            self.triggered_inferences = [InferenceStatus()]
                        else:
                            self.triggered_inferences.append(InferenceStatus()) # type: ignore
                        self.triggered_inferences[-1].workflow_id = workflow_id # type: ignore
                    except Exception as e:
                        logger.error(f'Error running workflow: {e}')
                    self.trigger_run_workflow = False





            #         if self.perform_analysis and not os.path.exists(csv_path):
            #             logger.info('Extracting defects...')
            #             from ifm_image_defect_detection.defectRecognition_toCSV import (
            #                 defect_recognition,
            #             )

            #             defect_recognition(
            #                 image_file.name, model_binary.name, model_classiciation.name
            #             )

            #         if not os.path.exists(csv_path):
            #             logger.warn(
            #                 'The csv file does not exist. Please (re)run the analysis '
            #                 'by checking "perform analysis" (again).'
            #             )
            #             continue

            #         # create result subsection
            #         analysis_entry = IFMAnalysisResult(file=csv_path)

            #         # read csv file and extract the defect prevalence
            #         defect_data = pd.read_csv(csv_path, skiprows=2)
            #         defect_columns = ['Whiskers', 'Chipping', 'Scratch', 'No Error']
            #         defect_data['type'] = defect_data[defect_columns].idxmax(axis=1)
            #         relative_share = defect_data['type'].value_counts(normalize=True)

            #         analysis_entry.defect_prevalence = DefectPrevalence(
            #             whiskers=relative_share.get('Whiskers', 0),
            #             chipping=relative_share.get('Chipping', 0),
            #             scratch=relative_share.get('Scratch', 0),
            #             no_error=relative_share.get('No Error', 0),
            #         )

            #         # add the result to the analysis output and update the workflow
            #         self.outputs.append(analysis_entry)
            #         archive.workflow2.outputs.append(
            #             Link(
            #                 name='Extracted Features',
            #                 section=analysis_entry,
            #             )
            #         )

            #         # create plot
            #         defect_mapping = {
            #             'Whiskers': 1,
            #             'Chipping': 2,
            #             'Scratch': 3,
            #             'No Error': 4,
            #         }
            #         defect_data['label'] = defect_data['type'].map(defect_mapping)

            #         heatmap = go.Heatmap(
            #             x=defect_data['x'],
            #             y=defect_data['y'],
            #             z=defect_data['label'],
            #             colorscale='Viridis',
            #             colorbar=dict(
            #                 tickvals=[1, 2, 3, 4],
            #                 ticktext=defect_columns,
            #                 title='Defect Type',
            #             ),
            #         )

            #         figure = go.Figure(data=heatmap)
            #         figure.update_layout(
            #             title='Heatmap of Defect Distribution',
            #             xaxis_title='X Position',
            #             yaxis_title='Y Position',
            #             xaxis=dict(scaleanchor='y'),
            #             yaxis=dict(scaleanchor='x'),
            #             autosize=True,
            #         )

            #         figure_json = figure.to_plotly_json()
            #         figure_json['config'] = {'staticPlot': True}
            #         self.figures.append(
            #             PlotlyFigure(
            #                 label='Defect Distribution Heatmap',
            #                 index=0,
            #                 figure=figure_json,
            #             )
            #         )

            # self.perform_analysis = False


m_package.__init_metainfo__()
