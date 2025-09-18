from nomad.config.models.ui import (
    App,
    Column,
    Format,
    ModeEnum,
    SearchQuantities,
)

schema = 'nomad_uibk_plugin.schema_packages.IFMschema.IFMTwoStepAnalysisResult'

ifm_analysis_app = App(
    label='IFM Analysis',
    path='ifm-analysis',
    category='UIBK',
    description='Search results of IFM Two Step Analysis',
    readme="""
        This app allows you to search for the results of IFM Two Step Analysis
        within NOMAD.
    """,
    search_quantities=SearchQuantities(include=[f'*#{schema}']),
    filters_locked={'entry_type': 'IFMTwoStepAnalysisResult'},
    columns=[
        Column(
            search_quantity=f'data.name#{schema}',
            selected=True,
            title='Entry name',
        ),
        Column(
            search_quantity=f'data.sample.lab_id#{schema}',
            selected=True,
            title='Sample ID',
        ),
        Column(
            search_quantity=f"data.defect_prevalence[?name=='No Error'].prevalence#{schema}",  # noqa: E501
            selected=True,
            title='No defect - area fraction',
            format=Format(decimals=5, mode=ModeEnum.STANDARD),
        ),
        Column(
            search_quantity=f"data.defect_prevalence[?name=='Whiskers'].prevalence#{schema}",  # noqa: E501
            selected=True,
            title='Whiskers - area fraction',
            format=Format(decimals=5, mode=ModeEnum.STANDARD),
        ),
        Column(
            search_quantity=f"data.defect_prevalence[?name=='Chipping'].prevalence#{schema}",  # noqa: E501
            selected=True,
            title='Chipping - area fraction',
            format=Format(decimals=5, mode=ModeEnum.STANDARD),
        ),
        Column(
            search_quantity=f"data.defect_prevalence[?name=='Scratch'].prevalence#{schema}",  # noqa: E501
            selected=True,
            title='Scratches - area fraction',
            format=Format(decimals=5, mode=ModeEnum.STANDARD),
        ),
        Column(search_quantity='entry_name', title='Name'),
        Column(search_quantity='entry_type'),
        Column(search_quantity='upload_create_time', title='Upload time'),
        Column(search_quantity='entry_create_time', title='Entry creation time'),
        Column(search_quantity='authors', title='Upload authors'),
        Column(search_quantity='comment'),
        Column(search_quantity='datasets'),
        Column(search_quantity='published', title='Access'),
    ],
)
