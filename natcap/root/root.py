"""ROOT InVEST Model."""

import csv
import logging
import multiprocessing
import os
import sys
from math import sqrt

import pandas as pd
import pygeoprocessing
import PySide2  # pragma: no cover
from natcap.invest import spec_utils
from natcap.invest import validation
from natcap.invest.ui import inputs
from natcap.invest.ui import model
from natcap.root import __version__
from natcap.root import arith_parser as ap
from natcap.root import optimization
from natcap.root import postprocessing
from natcap.root import preprocessing
from natcap.root import rootcore
from osgeo import gdal
from osgeo import ogr
from qtpy import QtGui
from qtpy import QtWidgets

LOGGER = logging.getLogger(__name__)

try:
    QApplication = QtGui.QApplication
except AttributeError:
    QApplication = QtWidgets.QApplication

APP = QApplication.instance()
if APP is None:
    APP = QApplication([])  # pragma: no cover


class RootInputError(Exception):
    pass


ARGS_SPEC = {
    'model_name': 'ROOT',
    'module': __name__,
    'userguide_html': '../documentation/root.html',
    'args_with_spatial_overlap': {
        "spatial_keys": [],
    },
    'args': {
        'workspace_dir': spec_utils.WORKSPACE,
        'results_suffix': spec_utils.SUFFIX,
        'do_preprocessing': {
            'type': 'boolean',
            'required': True,
            'about': (
                "Check to create marginal value tables based on "
                "raster/serviceshed inputs"),
            'name': "Do Preprocessing",
        },
        'activity_mask_table_path': {
            'type': 'csv',
            'required': False,
            'about': (
                "Table with paths for activity masks. See User's Guide."),
            'name': 'Activity Mask Table (CSV)',
        },
        'impact_raster_table_path': {
            'type': 'csv',
            'required': True,
            'about': (
                "Table that lists names and filepaths for impact "
                "potential rasters.<br><br>ROOT will aggregate these "
                "rasters to the spatial units outlined by the SDU "
                "grid.  See User's Guide for details on file format."),
            'name': 'Impact Potential Raster Table (CSV)',
        },
        'serviceshed_shapefiles_table': {
            'type': 'csv',
            'required': True,
            'about': (
                "Table that lists names, uris, and service value "
                "columns for servicesheds.<br><br>ROOT will calculate "
                "the overlap area and weighted average of each "
                "serviceshed for each SDU. See User's Guide for "
                "details on file format."),
            'name': 'Spatial Weighting Maps Table (CSV)',
        },
        'combined_factor_table_path': {
            'type': 'csv',
            'required': True,
            'about': (
                "This table allows the user to construct composite "
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
                "serviceshed shapefile."),
            'name': 'Composite Factor Table (CSV)',
        },
        'spatial_decision_unit_shape': {
            'type': 'freestyle_string',
            'required': True,
            'about': (
                "Determines the shape of the SDUs used to aggregate "
                "the impact potential and spatial weighting maps.  Can "
                "be square or hexagon to have ROOT generate an "
                "appropriately shaped regular grid, or can be the full "
                "path to an existing SDU shapefile.  If an existing "
                "shapefile is used, it must contain a unique ID field "
                "SDU_ID."),
            'name': 'Spatial Decision Unit Shape',
        },
        'spatial_decision_unit_area': {
            'type': 'number',
            'required': False,
            'about': (
                "Area of each grid cell in the constructed grid. "
                "Measured in hectares (1 ha = 10,000m^2). This is "
                "ignored if an existing file is used as the SDU map."),
            'name': 'Spatial Decision Unit Area (ha)',
        },
        'aoi_file_path': {
            'type': 'vector',
            'required': False,
            'about': (
                "Area of interest outline. Used to mask spatial analysis."),
            'name': 'AOI shapefile (optional)',
        },
        'advanced_args_json_path': {
            'type': 'file',
            'required': False,
            'about': (
                "json file with advanced options"),
            'name': 'Advanced options (json)'
        },

        'do_optimization': {
            'type': 'boolean',
            'required': True,
            'about': "Check to perform optimization",
            'name': "Do Optimization",
        },
        'optimization_suffix': {
            'type': 'freestyle_string',
            'required': False,
            'about': (
                "This text will be appended to the optimization "
                "output folder to distinguish separate analyses."),
            'name': 'Optimization Results Suffix (Optional)',
        },
        'frontier_type': {
            'type': 'option_string',
            'required': "do_optimization",
            'about': (
                "Determines the mode of operation for the optimizer. "
                "Can be: <ul><li>weight_table: use to provide "
                "particular objective weights for a given number of "
                "optimization runs.  </li><li>frontier: evenly-spaced "
                "frontier (requires exactly two objectives in "
                "objectives table).</li> <li>n_dim_frontier: use for "
                "more than two objective analysis.  Randomly samples "
                "points on N dimensional "
                "frontier.</li><li>n_dim_outline: constructs 2D "
                "frontiers for all pairs of objectives.</li></ul>"),
            'name': 'Analysis Type',
            'options': {
                "weight_table": {"display_name": "Weight Table"},
                "frontier": {"display_name": "Frontier"},
                "n_dim_frontier": {
                    "display_name": "Sample N Dimensional Points on Frontier"},
                "n_dim_outline": {
                    "display_name": (
                        "Construct frontiers for all pairs of objectives")},
            },
        },
        'number_of_frontier_points': {
            'type': 'number',
            'required': "do_optimization",
            'about': (
                "Determines number of points to calculate.  Note that "
                "for 'frontier' runs, the actual number of points will "
                "be higher than this number to ensure whole-frontier "
                "coverage."),
            'name': 'Number of Frontier Points',
        },
        'objectives_table_path': {
            'type': 'csv',
            'required': "do_optimization",
            'about': (
                "This table identifies which factors from the "
                "preprocessing results to use as objectives in the "
                "optimization.  Values in the 'objective' column "
                "should match names assigned to marginal values, "
                "servicesheds, or combined factors.  For the 'single' "
                "optimization option, the values in 'weight' should be "
                "the appropriate numeric values.  For all others, the "
                "'weight' column should say 'minimize' or 'maximize' "
                "according to whether the optimization should prefer "
                "lower or higher values for that factor, respectively."),
            'name': 'Objectives Table (CSV)',
        },
        'targets_table_path': {
            'type': 'csv',
            'required': "do_optimization",
            'about': (
                "This table identifies targets (constraints) to apply "
                "to the optimization."),
            'name': 'Targets Table (CSV)',
            'validation_options': {
                'required_fields': ["name", "cons_type", "value"]
            }
        }
    }
}


def execute(args):
    """root.

    """
    LOGGER.info(f'Running ROOT version {__version__}')
    internal_args = rootcore.parse_args(args)

    # with open(os.path.join(internal_args['workspace'], 'root_args.json'), 'w') as root_args_file:
    #     json.dump(internal_args, root_args_file, indent=2)

    if args['do_preprocessing']:
        print('doing pre-processing')
        preprocessing.execute(internal_args)
    else:
        print('skipping pre-processing')

    if args['do_optimization']:
        print('doing optimization')
        optimization.execute(internal_args)
        postprocessing.execute(internal_args)
    else:
        print('skipping optimization')





def validate_sdu_shape_arg(arg_val):
    if arg_val == 'hexagon' or arg_val == 'square':
        return True
    elif not os.path.isfile(arg_val):
        msg = 'Error in SDU shape: invalid value "{}". '.format(arg_val)
        msg += '\nSpatial Decision Unit Shape must be square, hexagon, or a path to a shapefile.'
        raise RootInputError(msg)


def validate(args, limit_to=None):
    validation_warnings = validation.validate(
        args, ARGS_SPEC['args'], ARGS_SPEC['args_with_spatial_overlap'])

    invalid_keys = validation.get_invalid_keys(validation_warnings)

    if 'spatial_decision_unit_shape' not in invalid_keys:
        sdu_valid = True
        try:
            if not validate_sdu_shape_arg(args['spatial_decision_unit_shape']):
                sdu_valid = False
        except RootInputError:
            sdu_valid = False

        if not sdu_valid:
            validation_warnings.append(
                (['spatial_decision_unit_shape'],
                 ('Spatial Decision Unit Shape must be "square", "hexagon", '
                  'or a path to a vector')))

    return validation_warnings


def _create_input_kwargs_from_args_spec(args_key, validate=True):
    """Helper function to return kwargs for most model inputs.
    Args:
        args_key: The args key of the input from which a kwargs
            dict is being built.
        validate=True: Whether to include the ``validator`` key in the return
            kwargs dict.  Some inputs (e.g. ``Checkbox``) do not take a
            ``validator`` argument.

    Returns:
        A dict of ``kwargs`` to explode to an ``inputs.GriddedInput``
        object at creation time.
    """
    model_spec = ARGS_SPEC['args']
    kwargs = {
        'args_key': args_key,
        'helptext': model_spec[args_key]['about'],
        'label': model_spec[args_key]['name'],
    }

    arg_type = model_spec[args_key]['type']

    if validate and arg_type != 'option_string':
        kwargs['validator'] = validate

    if arg_type == 'option_string':
        spec_options = model_spec[args_key]['options']
        kwargs['options'] = [
            value['display_name'] for value in spec_options.values()]
        kwargs['return_value_map'] = {
            value['display_name']: key for (key, value) in
            spec_options.items()}

    return kwargs


class Root(model.InVESTModel):
    def __init__(self):
        model.InVESTModel.__init__(
            self,
            label='ROOT',
            target=execute,
            validator=validate,
            localdoc=u'../documentation/root.html'
        )

        # Explicitly setting the default workspace because the InVEST UI's
        # approach relies on self.target.__module__, which isn't reliable when
        # execute is in the same script as the launcher.  In this case, the
        # module name is __main__.  Technically true, but not user-readable.
        self.workspace.set_value(os.path.normpath(
            os.path.expanduser('~/Documents/root_workspace')))

        self.preprocessing_container = inputs.Container(
            args_key=u'preprocessing_container',
            expandable=False,
            expanded=True,
            label=u'Preprocessing Arguments')
        self.add_input(self.preprocessing_container)

        self.do_preprocessing = inputs.Checkbox(
            **_create_input_kwargs_from_args_spec(
                'do_preprocessing', validate=False))
        self.preprocessing_container.add_input(self.do_preprocessing)

        self.activity_mask_raster_path = inputs.File(
            **_create_input_kwargs_from_args_spec('activity_mask_table_path'))
        self.preprocessing_container.add_input(self.activity_mask_raster_path)

        self.impact_raster_table_path = inputs.File(
            **_create_input_kwargs_from_args_spec('impact_raster_table_path'))
        self.preprocessing_container.add_input(self.impact_raster_table_path)

        self.serviceshed_shapefiles_table = inputs.File(
            **_create_input_kwargs_from_args_spec('serviceshed_shapefiles_table'))
        self.preprocessing_container.add_input(self.serviceshed_shapefiles_table)

        self.combined_factor_table_path = inputs.File(
            **_create_input_kwargs_from_args_spec('combined_factor_table_path'))
        self.preprocessing_container.add_input(self.combined_factor_table_path)

        self.spatial_decision_unit_shape = inputs.Text(
            **_create_input_kwargs_from_args_spec('spatial_decision_unit_shape'))
        self.preprocessing_container.add_input(self.spatial_decision_unit_shape)

        self.spatial_decision_unit_area = inputs.Text(
            **_create_input_kwargs_from_args_spec('spatial_decision_unit_area'))
        self.preprocessing_container.add_input(self.spatial_decision_unit_area)

        self.aoi_file_path = inputs.Text(
            **_create_input_kwargs_from_args_spec('aoi_file_path'))
        self.preprocessing_container.add_input(self.aoi_file_path)

        self.advanced_args_json_path = inputs.Text(
            **_create_input_kwargs_from_args_spec('advanced_args_json_path'))
        self.preprocessing_container.add_input(self.advanced_args_json_path)


        self.optimization_container = inputs.Container(
            args_key=u'optimization_container',
            expandable=False,
            expanded=True,
            label=u'Optimization Arguments')
        self.add_input(self.optimization_container)
        self.do_optimization = inputs.Checkbox(
            **_create_input_kwargs_from_args_spec(
                'do_optimization', validate=False))
        self.optimization_container.add_input(self.do_optimization)

        self.optimization_suffix = inputs.Text(
            **_create_input_kwargs_from_args_spec('optimization_suffix'))
        self.optimization_container.add_input(self.optimization_suffix)

        self.frontier_type = inputs.Dropdown(
            **_create_input_kwargs_from_args_spec('frontier_type'))
        self.optimization_container.add_input(self.frontier_type)

        self.number_of_frontier_points = inputs.Text(
            **_create_input_kwargs_from_args_spec('number_of_frontier_points'))
        self.optimization_container.add_input(self.number_of_frontier_points)

        self.objectives_table_path = inputs.File(
            **_create_input_kwargs_from_args_spec('objectives_table_path'))
        self.optimization_container.add_input(self.objectives_table_path)

        self.targets_table_path = inputs.File(
            **_create_input_kwargs_from_args_spec('targets_table_path'))
        self.optimization_container.add_input(self.targets_table_path)

    def assemble_args(self):
        args = {
            self.workspace.args_key: self.workspace.value(),
            self.suffix.args_key: self.suffix.value(),
            self.preprocessing_container.args_key: self.preprocessing_container.value(),
            self.optimization_container.args_key: self.optimization_container.value(),
        }
        if self.optimization_suffix.value():
            args[self.optimization_suffix.args_key] = self.optimization_suffix.value()
        if self.preprocessing_container.value():
            args[self.do_preprocessing.args_key] = self.do_preprocessing.value()
            args[self.activity_mask_raster_path.args_key] = self.activity_mask_raster_path.value()
            args[self.impact_raster_table_path.args_key] = self.impact_raster_table_path.value()
            args[self.serviceshed_shapefiles_table.args_key] = self.serviceshed_shapefiles_table.value()
            args[self.combined_factor_table_path.args_key] = self.combined_factor_table_path.value()
            args[self.spatial_decision_unit_shape.args_key] = self.spatial_decision_unit_shape.value()
            args[self.spatial_decision_unit_area.args_key] = self.spatial_decision_unit_area.value()
            args[self.aoi_file_path.args_key] = self.aoi_file_path.value()
            args[self.advanced_args_json_path.args_key] = self.advanced_args_json_path.value()

        if self.optimization_container.value():
            args[self.do_optimization.args_key] = self.do_optimization.value()
            args[self.frontier_type.args_key] = self.frontier_type.value()
            args[self.number_of_frontier_points.args_key] = self.number_of_frontier_points.value()
            args[self.objectives_table_path.args_key] = self.objectives_table_path.value()
            args[self.targets_table_path.args_key] = self.targets_table_path.value()

        return args


def main():
    multiprocessing.freeze_support()

    if '--test-imports' in sys.argv:
        sys.exit(0)

    if '-v' in sys.argv:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    # logging.basicConfig will by default write all streams to stderr.
    logging.basicConfig(level=log_level)

    LOGGER.info('Constructing UI instance')
    ui = Root()

    LOGGER.info('Adjusting window and setting up connections.')
    ui.run(quickrun=False)

    LOGGER.info('Entering event loop.')
    _ = APP.exec_()
    LOGGER.info('Exiting.')


if __name__ == '__main__':
    main()
