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
from nomad_pvcomb.schema_packages.activities import File
from nomad_pvcomb.schema_packages.processes import (
    SolarCellJVCurve,
    SolarCellJVCurveDark,
)

from nomad_uibk_plugin.schema_packages.JVschema import UIBK_JVMeasurement
from nomad_uibk_plugin.schema_packages.sample import UIBKSampleReference
from nomad_uibk_plugin.utils import safe_float

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

    def match_sample_name(self, label: str):
        """
        Split measurement curve label and match it into sample id, prefix, position
        Return None as sample id (generated_lab_id) if not matched to any pattern
        """
        split_label = re.compile(r'^(.*)_(\d+)$').match(label)
        new_sample_name_no_position = split_label.group(1)  # pyright: ignore[reportOptionalMemberAccess]
        new_position = split_label.group(2)  # pyright: ignore[reportOptionalMemberAccess]

        # check if the sample is supported. Only types e and eZ for now
        match_e = re.compile(r'^(\d{8})_(\d+)e(?:-(\d+))?$').match(
            new_sample_name_no_position
        )
        match_eZ = re.compile(r'^(\d{8})_(\d+)eZ(?:-(\d+))?$').match(
            new_sample_name_no_position
        )
        if match_e:
            if match_e.group(3):
                generated_lab_id = (
                    f'{match_e.group(1)}_{match_e.group(2)}-{match_e.group(3)}'
                )
            else:
                generated_lab_id = f'{match_e.group(1)}_{match_e.group(2)}-1'
            new_prefix = 'e'
        elif match_eZ:
            if match_eZ.group(3):
                generated_lab_id = (
                    f'{match_eZ.group(1)}_{match_eZ.group(2)}-{match_eZ.group(3)}'
                )
            else:
                generated_lab_id = f'{match_eZ.group(1)}_{match_eZ.group(2)}-1'
            new_prefix = 'eZ'
        else:
            generated_lab_id = None
            new_prefix = None

        return generated_lab_id, new_prefix, new_position

    def parse(  # noqa: PLR0912, PLR0915
        self,
        mainfile: str,
        archive: 'EntryArchive',
        logger: 'BoundLogger',
    ) -> None:
        logger.info('JVParser.parse')
        archive.metadata.entry_type = 'RawJVMeasurementFile'

        with open(mainfile) as file:
            source_json = json.load(file)

        entries = []
        for measurement in source_json:
            source_data = measurement['dataStorage']['data']
            generated_lab_id, new_prefix, new_position = self.match_sample_name(
                source_data['label']
            )
            if generated_lab_id is None:
                continue

            active_area = safe_float(source_data['activeArea'])
            if active_area is None:
                logger.warning(
                    f'entry {source_data["label"]} skipped due to missing active area'
                )
                continue
            else:
                active_area_cm2 = active_area / 100  # to cm^2

            light_intensity = safe_float(source_data['powerInput'])
            if light_intensity is None:
                logger.warning(
                    f'entry {source_data["label"]} skipped due to missing light '
                    + 'intensity'
                )
                continue
            else:
                light_intensity_nomad_unit = light_intensity / 10  # to mW/cm**2
            try:
                jv_curve = SolarCellJVCurve(
                    label_name=source_data['measurementInfo']['lightId'],
                    datetime=source_data['measurementInfo']['lightTime'],
                    cell_id=new_position,
                    active_area=active_area_cm2,
                    cell_name=source_data['label'],
                    current_density=np.array(source_data['calculationValues']['iLight'])
                    * 1000
                    / active_area_cm2,  # conversion to mA/cm^2
                    voltage=source_data['calculationValues']['uLight'],
                    light_intensity=light_intensity_nomad_unit,
                    open_circuit_voltage=safe_float(source_data['voc']),
                    short_circuit_current_density=safe_float(source_data['jsc']),
                    fill_factor_in_percent=safe_float(source_data['fF']),
                    efficiency_in_percent=safe_float(source_data['eff']),
                    potential_at_maximum_power_point=safe_float(source_data['mppU']),
                    current_density_at_maximum_power_point=safe_float(
                        source_data['mppI']
                    ),  # divide by area later if not None
                    series_resistance=safe_float(source_data['rs']),
                    shunt_resistance=safe_float(source_data['rp']),
                )
            except Exception as e:
                logger.warning(f'entry {source_data["label"]} skipped due to {e}')
                continue
            if jv_curve.current_density_at_maximum_power_point:
                jv_curve.current_density_at_maximum_power_point = (
                    jv_curve.current_density_at_maximum_power_point / active_area_cm2
                )  # pyright: ignore[reportOperatorIssue]

            try:
                dark_jv_curve = SolarCellJVCurveDark(
                    label_name=source_data['measurementInfo']['darkId'],
                    datetime=source_data['measurementInfo']['darkTime'],
                    cell_id=new_position,
                    active_area=active_area_cm2,
                    cell_name=source_data['label'],
                    current_density=np.array(source_data['calculationValues']['iDark'])
                    * 1000
                    / active_area_cm2,  # conversion to mA/cm^2
                    voltage=source_data['calculationValues']['uDark'],
                    series_resistance=safe_float(source_data['darkRs']),
                    shunt_resistance=safe_float(source_data['darkRp']),
                )
            except Exception as e:
                logger.warning(f'entry {source_data["label"]} skipped due to {e}')
                continue

            entry_old_found = False
            for entry_old in entries:
                if (generated_lab_id == entry_old.samples[0].lab_id) and (
                    new_prefix == entry_old.samples[0].prefix
                ):
                    if new_position in entry_old.samples[0].position.split(', '):
                        logger.warning(
                            f'json entry {source_data["label"]} corresponds to a '
                            + f'measurement already recorded {entry_old.name}, '
                            + 'skipping...'
                        )
                        entry_old_found = True
                        break
                    entry_old.jv_curves.append(jv_curve)
                    entry_old.dark_jv_curves.append(dark_jv_curve)
                    entry_old.samples[0].position = (
                        entry_old.samples[0].position + ', ' + f'{new_position}'
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
                        lab_id=generated_lab_id,
                        position=f'{new_position}',
                        prefix=new_prefix,
                    )
                ]
                entry.files = File(data_files=[mainfile.split('/raw/')[-1]])
                entries.append(entry)

        for entry in entries:
            file_name = (
                f'JV_{entry.samples[0].lab_id}-{entry.samples[0].prefix}.archive.json'
            )
            create_archive(
                entity=entry,
                archive=archive,
                file_name=file_name,
                overwrite=True,
            )
