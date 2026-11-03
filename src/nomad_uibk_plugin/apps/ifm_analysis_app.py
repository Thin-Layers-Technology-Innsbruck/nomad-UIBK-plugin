from nomad.config.models.ui import (
    App,
    Axis,
    AxisQuantity,
    AxisScale,
    Column,
    Dashboard,
    Format,
    Layout,
    Menu,
    MenuItemHistogram,
    MenuItemTerms,
    MenuSizeEnum,
    ModeEnum,
    ScaleEnum,
    SearchQuantities,
    WidgetBoxPlot,
    WidgetTerms,
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
            search_quantity=f"data.defect_prevalence[?name=='No Defect'].prevalence#{schema}",  # noqa: E501
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
            search_quantity=f"data.defect_prevalence[?name=='Scratching'].prevalence#{schema}",  # noqa: E501
            selected=True,
            title='Scratches - area fraction',
            format=Format(decimals=5, mode=ModeEnum.STANDARD),
        ),
        Column(
            search_quantity=f'data.image_masked#{schema}',
            selected=True,
            title='Masked automatically during analysis',
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
                        search_quantity=f'data.sample.lab_id#{schema}',
                        title='Sample ID',
                        options=10,
                    ),
                    MenuItemTerms(
                        search_quantity=f'data.image_masked#{schema}',
                        title='Masked automatically during analysis',
                        options=3,
                        show_input=False,
                    ),
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
    # dashboard = Dashboard.model_validate({'widgets': yaml.safe_load(widgets_yaml)})
    dashboard=Dashboard(
        widgets=[
            WidgetBoxPlot(
                autorange=True,
                subgroup_by=None,
                group_by_size=10,
                group_by=f'data.defect_prevalence[*].name#{schema}',
                show_points=True,
                sample_size=1000,
                y=AxisQuantity(
                    search_quantity=f'data.defect_prevalence[*].count#{schema}',
                    unit='count',
                ),
                layout={
                    'xxl': Layout(minH=3, minW=3, h=8, w=24, y=0, x=0),
                    'xl': Layout(minH=3, minW=3, h=7, w=21, y=0, x=0),
                    'lg': Layout(minH=3, minW=3, h=6, w=17, y=0, x=0),
                    'md': Layout(minH=3, minW=3, h=5, w=12, y=0, x=0),
                    'sm': Layout(minH=3, minW=3, h=4, w=8, y=0, x=0),
                },
            ),
            WidgetTerms(
                scale='linear',
                search_quantity=f'data.image_masked#{schema}',
                layout={
                    'xxl': Layout(minH=3, minW=3, h=8, w=6, y=0, x=24),
                    'xl': Layout(minH=3, minW=3, h=7, w=6, y=0, x=21),
                    'lg': Layout(minH=3, minW=3, h=6, w=5, y=0, x=17),
                    'md': Layout(minH=3, minW=3, h=5, w=4, y=0, x=12),
                    'sm': Layout(minH=3, minW=3, h=4, w=3, y=0, x=8),
                },
            ),
        ]  # pyright: ignore[reportArgumentType]
    ),
)
