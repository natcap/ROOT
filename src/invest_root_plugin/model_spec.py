from natcap.invest import spec
from natcap.invest.unit_registry import u

MODEL_SPEC = spec.ModelSpec(
    model_id="root_plugin",
    model_title="ROOT",
    userguide="https://natcap.github.io/ROOT/",
    module_name="invest_root_plugin",
    input_field_order=[
        [
            "workspace_dir", "results_suffix",
        ],
        [
            "do_preprocessing", "activity_mask_table_path",
            "impact_raster_table_path", "serviceshed_shapefiles_table",
            "combined_factor_table_path", "spatial_decision_unit_shape",
            "spatial_decision_unit_area", "aoi_file_path",
            "advanced_args_json_path",
        ],
        [
            "do_optimization", "optimization_suffix",
            "frontier_type", "number_of_frontier_points",
            "objectives_table_path", "targets_table_path"
        ]
    ],
    inputs=[
        spec.WORKSPACE,
        spec.SUFFIX,
        spec.N_WORKERS,
        spec.BooleanInput(
            id="do_preprocessing",
            name="Do Preprocessing",
            about="Check to create marginal value tables based on raster/serviceshed inputs",
        ),
        spec.FileInput(
            id="activity_mask_table_path",
            name="Activity Mask Table (CSV)",
            about="Table with paths for activity masks. See User's Guide.",
            required=False
        ),
        spec.FileInput(
            id="impact_raster_table_path",
            name="Impact Potential Raster Table (CSV)",
            about="Table that lists names and filepaths for impact "
                "potential rasters.<br /><br />ROOT will aggregate these "
                "rasters to the spatial units outlined by the SDU "
                "grid.  See User's Guide for details on file format.",
            required=False
        ),
        spec.FileInput(
            id="serviceshed_shapefiles_table",
            name="Spatial Weighting Maps Table (CSV)",
            about="Table that lists names, uris, and service value "
                "columns for servicesheds.<br /><br />ROOT will calculate "
                "the overlap area and weighted average of each "
                "serviceshed for each SDU. See User's Guide for "
                "details on file format.",
            required=False
        ),
        spec.FileInput(
            id="combined_factor_table_path",
            name="Composite Factor Table (CSV)",
            about="This table allows the user to construct composite "
                "factors, such as ecosystem services weighted by a "
                "serviceshed.  The value in the 'name' column will be "
                "used to identify this composite factor in summary "
                "tables, output maps, and in configuration tables in "
                "the optimization section.  The value in the 'factors' "
                "field should list which marginal values and "
                "shapefiles to multiply together to construct the "
                "composite factor.  The entry should be comma "
                "separated, such as 'sed_export, hydropower_capacity'. "
                "Use an '_' to identify fields of interest within a "
                "serviceshed shapefile.",
            required=True
        ),
        spec.StringInput(
            id="spatial_decision_unit_shape",
            name="Spatial Decision Unit Shape",
            about="Determines the shape of the SDUs used to aggregate "
                "the impact potential and spatial weighting maps.  Can "
                "be square or hexagon to have ROOT generate an "
                "appropriately shaped regular grid, or can be the full "
                "path to an existing SDU shapefile.  If an existing "
                "shapefile is used, it must contain a unique ID field "
                "SDU_ID.",
            required=False
        ),
        spec.NumberInput(
            id="spatial_decision_unit_area",
            name="Spatial Decision Unit Area (ha)",
            about="Area of each grid cell in the constructed grid. "
                "Measured in hectares (1 ha = 10,000m^2). This is "
                "ignored if an existing file is used as the SDU map.",
            required=False,
            units=u.hectare
        ),
        spec.FileInput(
            id="aoi_file_path",
            name="AOI shapefile (optional)",
            about="Area of interest outline. Used to mask spatial analysis.",
            required=False
        ),
        spec.FileInput(
            id="advanced_args_json_path",
            name="Advanced options (json)",
            about="json file with advanced options",
            required=False
        ),
        spec.BooleanInput(
            id="do_optimization",
            name="Do Optimization",
            about="Check to perform optimization",
        ),
        spec.StringInput(
            id="optimization_suffix",
            name="Optimization Results Suffix (Optional)",
            about="This text will be appended to the optimization "
                "output folder to distinguish separate analyses.",
            required=False
        ),
        spec.OptionStringInput(
            id="frontier_type",
            name="Analysis Type",
            options=[
                spec.Option(key="weight_table"),
                spec.Option(key="frontier"), 
                spec.Option(key="n_dim_frontier"),
                spec.Option(key="n_dim_outline")
            ],
            about="Determines the mode of operation for the optimizer. "
                "Can be: <ul><li>weight_table: use to provide "
                "particular objective weights for a given number of "
                "optimization runs.  </li><li>frontier: evenly-spaced "
                "frontier (requires exactly two objectives in "
                "objectives table).</li> <li>n_dim_frontier: use for "
                "more than two objective analysis.  Randomly samples "
                "points on N dimensional "
                "frontier.</li><li>n_dim_outline: constructs 2D "
                "frontiers for all pairs of objectives.</li></ul>",
            required="do_optimization"
        ),
        spec.NumberInput(
            id="number_of_frontier_points",
            name="Number of frontier points",
            about="Number of frontier points",
            required="do_optimization",
            units=None
        ),
        spec.FileInput(
            id="objectives_table_path",
            name="Objectives (CSV)",
            about="This table identifies objectives for the optimization",
            required="do_optimization"
        ),
        spec.FileInput(
            id="targets_table_path",
            name="Targets Table (CSV)",
            about="This table identifies targets (constraints) to apply "
                "to the optimization.",
            required="do_optimization"
        ),
    ],
    # All output paths are relative to the workspace dir
    outputs=[
        spec.SingleBandRasterOutput(
            id="sdu_grid_raster",
            about="rasterized version of sdu_grid shapefile with SDU_ID as value",
            created_if="do_preprocessing",
            path="sdu_grid.tif",
            data_type=int,
            units=None
        ),
        spec.TASKGRAPH_CACHE

        # spec.FileOutput(
        #     id="agreement_map_[]",
        #     about="Shows the number of times each SDU was selected.",
        #     created_if="do_optimization",
        #     path="optimization_results_[OPTIMIZATION_SUFFIX]/agreement_map.shp"
        # )
    ],
    # reporter='invest_root_plugin.reporter'
)