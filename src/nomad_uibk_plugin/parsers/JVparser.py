import json
import re
from typing import TYPE_CHECKING

import numpy as np
from nomad.config import config
from nomad.datamodel.data import EntryData
from nomad.datamodel.metainfo.annotations import ELNAnnotation
from nomad.metainfo import Quantity
from nomad.parsing.parser import MatchingParser
from nomad_measurements.utils import create_archive
from nomad_pvcomb.schema_packages.processes import (
    SolarCellJVCurve,
    SolarCellJVCurveDark,
)

from nomad_uibk_plugin.schema_packages.JVschema import UIBK_JVMeasurement
from nomad_uibk_plugin.schema_packages.sample import UIBKSampleReference

if TYPE_CHECKING:
    from nomad.datamodel.datamodel import EntryArchive
    from structlog.stdlib import BoundLogger

configuration = config.get_plugin_entry_point('nomad_uibk_plugin.parsers:jvjsonparser')


class RawFileJVMeasData(EntryData):
    """
    Section for an JV Measurements data .json file.
    """

    measurement = Quantity(
        type=UIBK_JVMeasurement,
        a_eln=ELNAnnotation(
            component='ReferenceEditQuantity',
        ),
    )


class JVParser(MatchingParser):
    """
    Parser for matching JV .json files and creating instances of UIBK_JVMeasurement.
    """

    def parse(
        self,
        mainfile: str,
        archive: 'EntryArchive',
        logger: 'BoundLogger',
    ) -> None:
        logger.info('JVParser.parse')
        data_file = mainfile.split('/raw/')[-1]
        archive.metadata.entry_type = 'RawJVMeasurementFile'
        entry = UIBK_JVMeasurement.m_from_dict(UIBK_JVMeasurement.m_def.a_template)  # pyright: ignore[reportArgumentType]

        with open(mainfile) as file:
            source_json = json.load(file)
            source_data = source_json['dataStorage']['data']
            split_label = re.compile(r'^(.*)_(\d+)$').match(source_data['label'])
        entry.jv_curves = [
            SolarCellJVCurve(
                label_name=source_data['measurementInfo']['lightId'],
                datetime=source_data['measurementInfo']['lightTime'],
                cell_id=split_label.group(2),  # pyright: ignore[reportOptionalMemberAccess]
                active_area=source_data['activeArea'] / 100,  # conversion to cm^2
                cell_name=source_data['label'],
                current_density=np.array(source_data['calculationValues']['iLight'])
                * 1000
                / source_data['activeArea']
                * 100,  # conversion to mA/cm^2
                voltage=source_data['calculationValues']['uLight'],
                light_intensity=source_data['powerInput'],
                open_circuit_voltage=source_data['voc'],
                short_circuit_current_density=source_data['jsc'],
                fill_factor_in_percent=source_data['fF'],
                efficiency_in_percent=source_data['eff'],
                potential_at_maximum_power_point=source_data['mppU'],
                current_density_at_maximum_power_point=source_data['mppI']
                / source_data['activeArea']
                * 100,  # conversion to mA/cm^2
                series_resistance=source_data['rs'],
                shunt_resistance=source_data['rp'],
            ),
        ]
        entry.dark_jv_curves = [
            SolarCellJVCurveDark(
                label_name=source_data['measurementInfo']['darkId'],
                datetime=source_data['measurementInfo']['darkTime'],
                cell_id=split_label.group(2),  # pyright: ignore[reportOptionalMemberAccess]
                active_area=source_data['activeArea'] / 100,  # conversion to cm^2
                cell_name=source_data['label'],
                current_density=np.array(source_data['calculationValues']['iDark'])
                * 1000
                / source_data['activeArea']
                * 100,  # conversion to mA/cm^2
                voltage=source_data['calculationValues']['uDark'],
                series_resistance=source_data['darkRs'],
                shunt_resistance=source_data['darkRp'],
            ),
        ]
        entry.samples = [
            UIBKSampleReference(
                lab_id=split_label.group(1),  # pyright: ignore[reportOptionalMemberAccess]
                position=split_label.group(2),  # pyright: ignore[reportOptionalMemberAccess]
            )
        ]

        file_str = ''.join(data_file.split('.')[:-1])
        file_name = f'{file_str}.archive.json'
        create_archive(
            entity=entry,
            archive=archive,
            file_name=file_name,
            overwrite=True,
        )
