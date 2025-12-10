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

    def parse(  # noqa: PLR0912, PLR0915
        self,
        mainfile: str,
        archive: 'EntryArchive',
        logger: 'BoundLogger',
    ) -> None:
        # TODO: make code less ugly, split in purposeful functions, reduce the use of
        # try/except, add comments
        logger.info('JVParser.parse')
        archive.metadata.entry_type = 'RawJVMeasurementFile'

        with open(mainfile) as file:
            source_json = json.load(file)

        entries = []
        for measurement in source_json:
            source_data = measurement['dataStorage']['data']
            split_label = re.compile(r'^(.*)_(\d+)$').match(source_data['label'])

            # check if the sample is supported. Only types e and eZ for now
            match_e = re.compile(r'^(\d{8})_(\d+)e(?:-(\d+))?$').match(
                split_label.group(1)  # pyright: ignore[reportOptionalMemberAccess]
            )
            match_eZ = re.compile(r'^(\d{8})_(\d+)eZ(?:-(\d+))?$').match(
                split_label.group(1)  # pyright: ignore[reportOptionalMemberAccess]
            )
            if match_e:
                if match_e.group(3):
                    generated_lab_id = (
                        f'{match_e.group(1)}_{match_e.group(2)}-{match_e.group(3)}'
                    )
                else:
                    generated_lab_id = f'{match_e.group(1)}_{match_e.group(2)}-1'
                new_position = f'e{split_label.group(2)}'  # pyright: ignore[reportOptionalMemberAccess]
            elif match_eZ:
                if match_eZ.group(3):
                    generated_lab_id = (
                        f'{match_eZ.group(1)}_{match_eZ.group(2)}-{match_eZ.group(3)}'
                    )
                else:
                    generated_lab_id = f'{match_eZ.group(1)}_{match_eZ.group(2)}-1'
                new_position = f'eZ{split_label.group(2)}'  # pyright: ignore[reportOptionalMemberAccess]
            else:
                continue

            try:
                active_area = (source_data['activeArea'] / 100,)  # conversion to cm^2
            except Exception:
                active_area = None
            try:
                current_density_light = (
                    np.array(source_data['calculationValues']['iLight'])
                    * 1000
                    / source_data['activeArea']
                    * 100
                )  # conversion to mA/cm^2
            except Exception:
                current_density_light = None
            try:
                current_density_at_maximum_power_point = (
                    source_data['mppI'] / source_data['activeArea'] * 100
                )
            except Exception:
                current_density_at_maximum_power_point = None
            try:
                jv_curve = SolarCellJVCurve(
                    label_name=source_data['measurementInfo']['lightId'],
                    datetime=source_data['measurementInfo']['lightTime'],
                    cell_id=split_label.group(2),  # pyright: ignore[reportOptionalMemberAccess]
                    active_area=active_area,
                    cell_name=source_data['label'],
                    current_density=current_density_light,
                    voltage=source_data['calculationValues']['uLight'],
                    light_intensity=source_data['powerInput'],
                    open_circuit_voltage=source_data['voc'],
                    short_circuit_current_density=source_data['jsc'],
                    fill_factor_in_percent=source_data['fF'],
                    efficiency_in_percent=source_data['eff'],
                    potential_at_maximum_power_point=source_data['mppU'],
                    current_density_at_maximum_power_point=current_density_at_maximum_power_point,
                    series_resistance=source_data['rs'],
                    shunt_resistance=source_data['rp'],
                )
            except Exception as e:
                logger.warning(f'entry {source_data["label"]} skipped due to {e}')
                continue
            try:
                current_density_dark = (
                    np.array(source_data['calculationValues']['iDark'])
                    * 1000
                    / source_data['activeArea']
                    * 100
                )  # conversion to mA/cm^2
            except Exception:
                current_density_dark = None
            try:
                dark_jv_curve = SolarCellJVCurveDark(
                    label_name=source_data['measurementInfo']['darkId'],
                    datetime=source_data['measurementInfo']['darkTime'],
                    cell_id=split_label.group(2),  # pyright: ignore[reportOptionalMemberAccess]
                    active_area=active_area,
                    cell_name=source_data['label'],
                    current_density=current_density_dark,
                    voltage=source_data['calculationValues']['uDark'],
                    series_resistance=source_data['darkRs'],
                    shunt_resistance=source_data['darkRp'],
                )
            except Exception as e:
                logger.warning(f'entry {source_data["label"]} skipped due to {e}')
                continue

            entry_old_found = False
            for entry_old in entries:
                if generated_lab_id == entry_old.samples[0].lab_id:  # pyright: ignore[reportOptionalMemberAccess]
                    entry_old.jv_curves.append(jv_curve)
                    entry_old.dark_jv_curves.append(dark_jv_curve)
                    entry_old.samples[0].position = (
                        entry_old.samples[0].position + ', ' + new_position
                    )
                    entry_old_found = True
                    break

            if not (entry_old_found):
                entry = UIBK_JVMeasurement.m_from_dict(
                    UIBK_JVMeasurement.m_def.a_template  # pyright: ignore[reportArgumentType]
                )
                entry.jv_curves = [jv_curve]
                entry.dark_jv_curves = [dark_jv_curve]
                entry.samples = [
                    UIBKSampleReference(
                        lab_id=generated_lab_id,  # pyright: ignore[reportOptionalMemberAccess]
                        position=new_position,  # pyright: ignore[reportOptionalMemberAccess]
                    )
                ]
                entries.append(entry)

        for entry in entries:
            file_name = f'JV_{entry.samples[0].lab_id}.archive.json'
            create_archive(
                entity=entry,
                archive=archive,
                file_name=file_name,
                overwrite=True,
            )
