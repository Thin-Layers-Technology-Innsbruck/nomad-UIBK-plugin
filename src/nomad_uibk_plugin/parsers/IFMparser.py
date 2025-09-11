import re
from typing import TYPE_CHECKING

from nomad.config import config
from nomad.datamodel.data import EntryData
from nomad.datamodel.metainfo.annotations import ELNAnnotation
from nomad.metainfo import Quantity
from nomad.parsing.parser import MatchingParser
from nomad_measurements.utils import create_archive

from nomad_uibk_plugin.schema_packages.IFMModelAndMeasurementSchema import (
    IFMMeasurement,
    IFMModel,
)

if TYPE_CHECKING:
    from nomad.datamodel.datamodel import EntryArchive
    from structlog.stdlib import BoundLogger

configuration = config.get_plugin_entry_point('nomad_uibk_plugin.parsers:xrfparser')


class RawFileIFMMeasData(EntryData):
    """
    Section for an IFM Measurements data file.
    """

    measurement = Quantity(
        type=IFMMeasurement,
        a_eln=ELNAnnotation(
            component='ReferenceEditQuantity',
        ),
    )


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
        data_file = mainfile.split('/')[-1]
        entry = IFMMeasurement.m_from_dict(IFMMeasurement.m_def.a_template)
        file_str = ''.join(data_file.split('.')[:-1])
        if data_file.split('.')[-1] == 'bmp':
            file_name = f'{file_str}.archive.json'
        else:
            file_str = re.sub(r'_info$', '', file_str)
            file_name = f'{file_str}.archive.json'
        create_archive(
            entity=entry,
            archive=archive,
            file_name=file_name,
            overwrite=True,
        )


class IFMModelParser(MatchingParser):
    """
    Parser for matching IFM .bmp or .xml files and creating instances of IFMMeasurement.
    """

    def parse(
        self,
        mainfile: str,
        archive: 'EntryArchive',
        logger: 'BoundLogger',
    ) -> None:
        logger.info('IFMModelParser.parse')
        data_file = mainfile.split('/')[-1]
        entry = IFMModel.m_from_dict(IFMModel.m_def.a_template)
        file_str = ''.join(data_file.split('.')[:-1])
        file_name = f'{file_str}.archive.json'
        create_archive(
            entity=entry,
            archive=archive,
            file_name=file_name,
            overwrite=True,
        )
