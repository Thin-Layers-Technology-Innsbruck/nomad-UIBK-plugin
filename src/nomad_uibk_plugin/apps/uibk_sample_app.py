from nomad.config.models.ui import (
    App,
    Axis,
    AxisScale,
    Column,
    Menu,
    MenuItemHistogram,
    MenuItemTerms,
    MenuSizeEnum,
    ScaleEnum,
    SearchQuantities,
)

schema = 'nomad_uibk_plugin.schema_packages.sample.UIBKSample'

uibk_sample_app = App(
    label='Samples',
    path='uibk-samples',
    category='UIBK',
    description='Search results for samples within UIBK definition',
    readme="""
        This app allows you to search for samples within UIBK definition.
    """,
    search_quantities=SearchQuantities(include=[f'*#{schema}']),
    filters_locked={'entry_type': 'UIBKSample'},
    columns=[
        Column(
            search_quantity=f'data.name#{schema}',
            selected=True,
            title='Entry name',
        ),
        Column(
            search_quantity=f'data.lab_id#{schema}',
            selected=True,
            title='Sample ID',
        ),
        Column(
            search_quantity=f'data.activities_performed.jv_measurement[*].name#{schema}',
            selected=True,
            title='JV measurements performed',
        ),
        Column(
            search_quantity=f'data.activities_performed.ifm_measurement[*].name#{schema}',
            selected=True,
            title='IFM measurements performed',
        ),
        Column(
            search_quantity=f'data.activities_performed.ifm_analysis[*].name#{schema}',
            selected=True,
            title='IFM analysis performed',
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
    menu=Menu(
        size=MenuSizeEnum.MD,
        title='Menu',
        items=[
            Menu(
                title='Sample',
                size=MenuSizeEnum.MD,
                items=[
                    MenuItemTerms(
                        search_quantity=f'data.lab_id#{schema}',
                        title='Sample ID',
                        options=10,
                    ),
                    # MenuItemTerms(
                    #     title='Types of activities performed',
                    #     search_quantity='search_quantities.path_archive',
                    #     options={
                    #         'data.activities_performed.jv_measurement.0.name': MenuItemOption(  # noqa: E501
                    #             label='JV Measurements',
                    #         ),
                    #         'data.activities_performed.ifm_measurement.0.name': MenuItemOption(  # noqa: E501
                    #             label='IFM Measurements',
                    #         ),
                    #         'data.activities_performed.ifm_analysis.0.name': MenuItemOption(  # noqa: E501
                    #             label='IFM Analysis',
                    #         ),
                    #     },
                    #     show_input=False,
                    # ),
                ],
            ),
            Menu(
                title='NOMAD Upload Information',
                size=MenuSizeEnum.MD,
                items=[
                    MenuItemTerms(
                        search_quantity='authors.name',
                        title='Upload author',
                        options=0,
                    ),
                    MenuItemHistogram(
                        x=Axis(
                            search_quantity='upload_create_time',
                            title='Upload Creation Time',
                        ),
                        y=AxisScale(
                            scale=ScaleEnum.LOG,
                        ),
                        title='Upload Creation Time',
                        show_input=True,
                        nbins=30,
                    ),
                ],
            ),
        ],
    ),
)
