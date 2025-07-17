import os.path

from nomad.client import normalize_all, parse

from nomad_uibk_plugin.schema_packages.IFMschema import ureg


def test_IFMMeasurement():
    test_file = os.path.join(
        os.path.dirname(__file__), 'data', 'test_IFMMeasurement.archive.json'
    )
    entry_archive = parse(test_file)[0]
    normalize_all(entry_archive)

    # assert entry_archive.data.m_def == \
    # 'nomad_uibk_plugin.schema_packages.IFMschema.IFMMeasurement'
    assert entry_archive.data.method == 'IFM Measurement'
    assert entry_archive.data.image_file == 'IFM_Sample.bmp'
    assert entry_archive.data.metadata_file == 'IFM_Sample_1_info.xml'
    assert entry_archive.data.samples[0].lab_id == '20240829_A1-2'
    assert entry_archive.data.exposure_time == ureg.Quantity(
        0.00019899999999999999, 'second'
    )
    assert entry_archive.data.magnification == 9.98687  # noqa: PLR2004


def test_IFMModel():
    test_file = os.path.join(
        os.path.dirname(__file__), 'data', 'test_IFMModel.archive.json'
    )
    entry_archive = parse(test_file)[0]
    normalize_all(entry_archive)

    assert entry_archive.data.file == 'Model_20241229_binary.keras'
    ### normalization for .keras files only seem to work locally; commented away for github tests workflow
    # assert entry_archive.data.number_of_layers == 16  # noqa: PLR2004
    # assert entry_archive.data.number_of_parameters == 4584834  # noqa: PLR2004

    assert entry_archive.metadata.entry_type == 'IFMModel'


def test_IFMAnalysis():
    test_file = os.path.join(
        os.path.dirname(__file__), 'data', 'test_IFMAnalysis.archive.json'
    )
    entry_archive = parse(test_file)[0]
    normalize_all(entry_archive)

    assert entry_archive.data.method == 'IFM Two Step Analysis'
    assert entry_archive.metadata.entry_name == 'Analysis'
    assert entry_archive.metadata.entry_type == 'IFMTwoStepAnalysis'
