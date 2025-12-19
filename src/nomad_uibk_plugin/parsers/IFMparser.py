from typing import TYPE_CHECKING

from nomad.config import config
from nomad.datamodel.data import EntryData
from nomad.datamodel.metainfo.annotations import ELNAnnotation
from nomad.metainfo import Quantity
from nomad.parsing.parser import MatchingParser

from nomad_uibk_plugin.schema_packages.IFMModelAndMeasurementSchema import (
    IFMMeasurement,
    IFMModel,
)

if TYPE_CHECKING:
    from nomad.datamodel.datamodel import EntryArchive
    from structlog.stdlib import BoundLogger

configuration = config.get_plugin_entry_point('nomad_uibk_plugin.parsers:ifmparser')


class IFMParser(MatchingParser):
    """
    Parser for matching IFM .bmp or .xml files and creating instances of IFMMeasurement.
    """

    def parse(
        self,
        mainfile: str,
        archive: 'EntryArchive',
        logger: 'BoundLogger',
    ) -> None:
        logger.info('IFMParser.parse')
        data_file = mainfile.split('/raw/')[-1]
        data_path = data_file.split('/')
        if len(data_path) > 1:
            folders_path = '/'.join(data_path[:-1]) + '/'
        else:
            folders_path = ''
        image_file = f'{folders_path}texture.bmp'
        metadata_file = f'{folders_path}info.xml'
        file_name = f'{folders_path}texture.archive.json'
        if data_file.split('.')[-1] == 'bmp':
            archive.metadata.entry_type = 'RawMeasurementFile'
            image_file_exists = True
            metadata_file_exists = archive.m_context.raw_path_exists(metadata_file)
        else:
            image_file_exists = archive.m_context.raw_path_exists(image_file)
            metadata_file_exists = True
            archive.metadata.entry_type = 'RawMeasurementMetadataFile'

        new_empty_entry = IFMMeasurement.m_from_dict(IFMMeasurement.m_def.a_template)
        new_empty_entry.method = 'IFM Measurement'

        reprocessing_needed = False

        with archive.m_context.update_entry(
            file_name, write=True, process=False
        ) as measurement_entry:
            if (
                measurement_entry.get('data') is None
                or measurement_entry['data'].get('method') != 'IFM Measurement'
            ):
                measurement_entry['data'] = new_empty_entry.m_to_dict(
                    with_root_def=True
                )
                logger.info(f'IFMMeasurement entry {file_name} created.')
                reprocessing_needed = True
            if (
                measurement_entry['data'].get('image_file') is None
                and image_file_exists
            ):
                measurement_entry['data']['image_file'] = image_file
                reprocessing_needed = True
            if (
                measurement_entry['data'].get('metadata_file') is None
                and metadata_file_exists
            ):
                measurement_entry['data']['metadata_file'] = metadata_file
                reprocessing_needed = True
        if reprocessing_needed:
            with archive.m_context.update_entry(
                file_name, write=True, process=True
            ) as measurement_entry:
                pass


configuration = config.get_plugin_entry_point(
    'nomad_uibk_plugin.parsers:ifmmodelparser'
)


class IFMModelParser(MatchingParser):
    """
    Parser for matching IFM .pt model and creating instances of IFMModel.
    """

    def parse(
        self,
        mainfile: str,
        archive: 'EntryArchive',
        logger: 'BoundLogger',
    ) -> None:
        logger.info('IFMModelParser.parse')
        data_file = mainfile.split('/raw/')[-1]
        file_name = f'{data_file[:-2]}archive.json'
        archive.metadata.entry_type = 'RawModelFile'
        new_empty_entry = IFMModel.m_from_dict(IFMModel.m_def.a_template)
        new_empty_entry.method = 'IFM Model'

        reprocessing_needed = False

        with archive.m_context.update_entry(
            file_name, write=True, process=False
        ) as model_entry:
            if (
                model_entry.get('data') is None
                or model_entry['data'].get('method') != 'IFM Model'
            ):
                model_entry['data'] = new_empty_entry.m_to_dict(with_root_def=True)
                logger.info(f'IFMModel entry {file_name} created.')
                reprocessing_needed = True
            if model_entry['data'].get('file') is None:
                model_entry['data']['file'] = data_file
                reprocessing_needed = True

        if reprocessing_needed:
            with archive.m_context.update_entry(
                file_name, write=True, process=True
            ) as model_entry:
                pass
