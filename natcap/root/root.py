"""ROOT InVEST Model."""

import os
import sys
import logging
import csv
import multiprocessing
from math import sqrt

import pandas as pd
from osgeo import ogr
import PySide2  # pragma: no cover
from qtpy import QtWidgets
from qtpy import QtGui
from natcap.invest.ui import model, inputs
from natcap.invest import validation

from natcap.root import __version__
from natcap.root import preprocessing
from natcap.root import postprocessing
from natcap.root import optimization

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
        "spatial_keys": ['potential_conversion_mask_path',
                         'spatial_decision_unit_shape'],
    },
    'args': {
        'workspace_dir': validation.WORKSPACE_SPEC,
        'results_suffix': validation.SUFFIX_SPEC,
        'do_preprocessing': {
            'type': 'boolean',
            'required': True,
            'about': (
                "Check to create marginal value tables based on "
                "raster/serviceshed inputs"),
            'name': "Do Preprocessing",
        },
        'marginal_raster_table_path': {
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
        'potential_conversion_mask_path': {
            'type': 'raster',
            'required': True,
            'about': (
                "Raster that indicates which pixels should be "
                "considered as potential activity locations.  Values "
                "must be 1 for activity locations or NODATA for "
                "excluded locations."),
            'name': 'Activity Mask Raster',
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
            'required': True,
            'about': (
                "Area of each grid cell in the constructed grid. "
                "Measured in hectares (1 ha = 10,000m^2). This is "
                "ignored if an existing file is used as the SDU map."),
            'name': 'Spatial Decision Unit Area (ha)',
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
            'validation_options': {
                'options': ["weight_table", "frontier", "n_dim_frontier",
                            "n_dim_outline"],
            }
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
    internal_args = parse_args(args)

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


def parse_args(ui_args):
    """
    Takes args from root.py InVEST UI, converts to the args expected by
    the optimization engine.

    Parses tables, also.

    :param ui_args:
    :return:
    """

    root_args = {}

    # rename some args
    # TODO: clean these up: remove redundancies and make consistent through UI and ROOT code
    root_args['workspace'] = ui_args['workspace_dir']
    root_args['baseline_file'] = 'baseline.csv'
    root_args['sdu_id_col'] = 'SDU_ID'

    if ui_args['do_preprocessing']:

        validate_raster_input_table(ui_args['marginal_raster_table_path'])
        validate_shapefile_input_table(ui_args['serviceshed_shapefiles_table'])
        validate_cft_table(ui_args['marginal_raster_table_path'],
                           ui_args['serviceshed_shapefiles_table'],
                           ui_args['combined_factor_table_path'])
        validate_sdu_shape_arg(ui_args['spatial_decision_unit_shape'])

        root_args['mask_raster'] = ui_args['potential_conversion_mask_path']
        root_args['grid_type'] = ui_args['spatial_decision_unit_shape']
        cell_area = ui_args['spatial_decision_unit_area'] * 10000
        if root_args['grid_type'] == 'square':
            root_args['cell_size'] = sqrt(cell_area)
        elif root_args['grid_type'] == 'hexagon':
            a = sqrt( (2*cell_area) / (3*sqrt(3)) )
            root_args['cell_size'] = 2 * a

        root_args['csv_output_folder'] = os.path.join(root_args['workspace'], 'sdu_value_tables')

        root_args['raster_table'] = _process_raster_table(ui_args['marginal_raster_table_path'])
        raster_table = root_args['raster_table']
        print('raster_table.activity_names: {}'.format(raster_table.activity_names))
        print('raster_table.factor_names: {}'.format(raster_table.factor_names))

        # process serviceshed table
        # TODO: this should be optional
        serviceshed_names = []  # descriptive name for the serviceshed
        serviceshed_list = []   # file paths for the .shps
        serviceshed_values = {} # name of the column in the .shp to use as weighting value (dict indexed by ss name)
        if 'serviceshed_shapefiles_table' in ui_args and os.path.isfile(ui_args['serviceshed_shapefiles_table']):
            with open(ui_args['serviceshed_shapefiles_table']) as tablefile:
                reader = csv.DictReader(tablefile)
                for row in reader:
                    serviceshed_names.append(row['name'])
                    serviceshed_list.append(row['file_path'])
                    serviceshed_values[row['name']] = [x.strip() for x in row['weight_col'].split(' ')]
            root_args['serviceshed_names'] = serviceshed_names
            root_args['serviceshed_list'] = serviceshed_list
            root_args['serviceshed_values'] = serviceshed_values
        else:
            root_args['serviceshed_names'] = None
            root_args['serviceshed_list'] = None
            root_args['serviceshed_values'] = None

        # combined_factors_table_path
        # TODO: this should be optional, too.
        combined_factors = {}
        if 'combined_factor_table_path' in ui_args and os.path.isfile(ui_args['combined_factor_table_path']):
            with open(ui_args['combined_factor_table_path'], 'r') as tablefile:
                reader = csv.DictReader(tablefile)
                for row in reader:
                    combined_factors[row['name']] = [x.strip() for x in row['factors'].split(' ')]
            root_args['combined_factors'] = combined_factors
        else:
            root_args['combined_factors'] = None

    if ui_args['do_optimization']:

        validate_objectives_and_constraints_tables(
            ui_args['objectives_table_path'],
            ui_args['targets_table_path'],
            ui_args['workspace_dir']
        )

        # some fixed vals
        root_args['use_linear_vars'] = True
        root_args['flatten_objectives'] = False
        root_args['result_table_name'] = 'optimization_solutions.csv'

        # simple renames
        root_args['analysis_type'] = ui_args['frontier_type']
        root_args['npts'] = int(ui_args['number_of_frontier_points'])

        # conditional vals
        if root_args['analysis_type'] == 'single':
            root_args['normalize_objectives'] = False
        else:
            root_args['normalize_objectives'] = True

        # sub-workspace directory and file names
        root_args['data_folder'] = os.path.join(root_args['workspace'], 'sdu_value_tables')

        # allow optional suffix to differentiate optimization runs
        if 'optimization_suffix' in ui_args and ui_args['optimization_suffix'] is not None:
            print('USING OPTIMIZATION SUFFIX: {}'.format(ui_args['optimization_suffix']))
            root_args['optimization_folder'] = os.path.join(root_args['workspace'],
                                                            'optimizations_{}'.format(ui_args['optimization_suffix']))
        else:
            root_args['optimization_folder'] = os.path.join(root_args['workspace'], 'optimizations')

        # process optimization config tables
        root_args['weights'] = _process_objectives_table(ui_args, root_args)
        root_args['constraints'] = _process_constraints_table(ui_args)

    return root_args


def _is_valid_float(str):
    try:
        float(str)
        return True
    except ValueError:
        return False


def _process_constraints_table(ui_args):
    constraints = []
    with open(ui_args['targets_table_path']) as tablefile:
        reader = csv.DictReader(tablefile)
        for row in reader:
            constraints.append((row['name'], row['cons_type'], float(row['value'])))
    return constraints


def _process_raster_table(filename):
    """
    TODO: refactor to preserve jsonability
        either dump class in favor of nested dict or write custom to/from json funcs

    returns a class with structure: raster_lookup[activity][factorname] = rasterpath
    behaves like a dictionary for lookup, but also has attributes
    rastertable.activity_names
    rastertable.factor_names

    :param filename:
    :return:
    """

    class RasterTable(object):
        def __init__(self, filepath):
            self.filepath = filepath
            self.raster_lookup = {}
            self.factor_names = []
            with open(filename, 'r') as tablefile:
                header = tablefile.readline()
                header_fields = [f.strip() for f in header.split(',')]
                assert header_fields[0] == 'name'
                self.activity_names = header_fields[1:]
                for activity in self.activity_names:
                    self.raster_lookup[activity] = {}
                for row in tablefile:
                    row_fields = [f.strip() for f in row.split(',')]
                    factor_name = row_fields[0]
                    self.factor_names.append(factor_name)
                    for activity, filepath in zip(self.activity_names, row_fields[1:]):
                        self.raster_lookup[activity][factor_name] = filepath

        def __getitem__(self, item):
            return self.raster_lookup[item]

        def __repr__(self):
            return self.filepath

    return RasterTable(filename)


def _process_objectives_table(ui_args, root_args):
    """
    There are two distinct cases based on the analysis type:
    weight_table: for this analysis, we just pass the filename on as the weights
    others: these will have provided a table with mins/maxs that we parse
        return a dict {factorname: +/- 1.0, ...}

    :param ui_args:
    :param root_args:
    :return:
    """

    if root_args['analysis_type'] == 'weight_table':
        return ui_args['objectives_table_path']
    else:
        optimization_objectives = {}
        min_choices = ['Minimize', 'minimize', 'Min', 'min', 'Minimum', 'minimum']
        max_choices = ['Maximize', 'maximize', 'Max', 'max', 'Maximum', 'maximum']
        with open(ui_args['objectives_table_path'], 'r') as tablefile:
            var_names = tablefile.readline().strip().split(',')
            minmax = tablefile.readline().strip().split(',')
            for v, mm in zip(var_names, minmax):
                if mm in min_choices:
                    optimization_objectives[v] = -1.0
                elif mm in max_choices:
                    optimization_objectives[v] = 1.0
                else:
                    LOGGER.warning('invalid "weight" value: {}'.format(mm))
                    raise ValueError('invalid "weight" value: {}'.format(mm))

        return optimization_objectives


def validate_raster_input_table(raster_table_path):
    rt = pd.read_csv(raster_table_path)
    not_found = []
    for c in rt.columns[1:]:
        for f in rt[c]:
            if not os.path.isfile(f):
                not_found.append(f)
    if len(not_found) > 0:
        msg = "Error in IPR table: the following rasters could not be located. Please check the filepaths:\n"
        for f in not_found:
            msg += "\t{}\n".format(f)
        raise RootInputError(msg)


def validate_shapefile_input_table(shapefile_table_path):
    table = pd.read_csv(shapefile_table_path)

    # check columns are correct
    correct_cols = ['name', 'file_path', 'weight_col']
    for t, c in zip(table.columns, correct_cols):
        if t != c:
            raise RootInputError('Error in SWM table: columns must be named {}'.format(', '.join(correct_cols)))

    # check all files exist
    not_found = []
    for f in table['file_path']:
        if not os.path.isfile(f):
            not_found.append(f)
    if len(not_found) > 0:
        msg = "Error in SWM table: the following shapefiles could not be located. Please check the filepaths:\n"
        for f in not_found:
            msg += "\t{}\n".format(f)
        raise RootInputError(msg)

    # check that the columns exist
    cols_not_found = []
    for _, row in table.iterrows():
        ds = ogr.Open(row['file_path'])
        lyr = ds.GetLayer()
        field_names = [field.name for field in lyr.schema]
        weight_cols = row['weight_col'].split(' ')
        for wc in weight_cols:
            if wc not in field_names:
                cols_not_found.append(wc)
    if len(cols_not_found) > 0:
        msg = "Error in SWM table: The following weight cols could not be found in the shapefile: {}".format(
            cols_not_found
        )
        raise RootInputError(msg)


def validate_sdu_shape_arg(arg_val):
    if arg_val == 'hexagon' or arg_val == 'square':
        return True
    elif not os.path.isfile(arg_val):
        msg = 'Error in SDU shape: invalid value "{}". '.format(arg_val)
        msg += '\nSpatial Decision Unit Shape must be square, hexagon, or a path to a shapefile.'
        raise RootInputError(msg)


def validate_cft_table(rt_path, st_path, cft_path):
    rt = pd.read_csv(rt_path)
    st = pd.read_csv(st_path)
    cft = pd.read_csv(cft_path)

    raster_factors = list(rt['name'])
    shape_factors = []
    for _, row in st.iterrows():
        sname = row['name']
        shape_factors.append(sname)
        wcols = row['weight_col'].split(' ')
        for wc in wcols:
            shape_factors.append('{}_{}'.format(sname, wc))
    all_factors = raster_factors + shape_factors

    invalid_factors = []
    for _, row in cft.iterrows():
        factors = row['factors'].split(' ')
        for f in factors:
            if f not in all_factors:
                try:
                    float(f)
                    continue
                except ValueError:
                    invalid_factors.append(f)

    if len(invalid_factors) > 0:
        msg = "Error in CF table: invalid factors found: {}".format(invalid_factors)
        raise RootInputError(msg)


def validate_objectives_and_constraints_tables(obj_table_file, cons_table_file, workspace):
    baseline_sdu_stats_file = os.path.join(workspace, 'sdu_value_tables', 'baseline.csv')
    with open(baseline_sdu_stats_file) as f:
        header_row = f.readline().strip()
        factors = header_row.split(',')

    # check objectives table:
    with open(obj_table_file) as f:
        header_row = f.readline().strip()
        objectives = header_row.split(',')
        not_found = []
        for obj in objectives:
            if obj not in factors:
                not_found.append(obj)
        if len(not_found) > 0:
            msg = "Error in Objectives table. The following factors were not found in the sdu value tables: {}".format(
                not_found
            )
            raise RootInputError(msg)

    # check constraints table:
    with open(cons_table_file) as f:
        header_row = f.readline()  #skip header
        not_found = []
        for row in f:
            row_vals = row.strip().split(',')
            if row_vals[0] not in factors:
                not_found.append(row_vals[0])
        if len(not_found) > 0:
            msg = "Error in Targets table. The following factors were not found in the sdu value tables: {}".format(
                not_found
            )
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

    if validate:
        kwargs['validator'] = validate

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

        self.marginal_raster_table_path = inputs.File(
            **_create_input_kwargs_from_args_spec('marginal_raster_table_path'))
        self.preprocessing_container.add_input(self.marginal_raster_table_path)


        self.serviceshed_shapefiles_table = inputs.File(
            **_create_input_kwargs_from_args_spec('serviceshed_shapefiles_table'))
        self.preprocessing_container.add_input(self.serviceshed_shapefiles_table)


        self.combined_factor_table_path = inputs.File(
            **_create_input_kwargs_from_args_spec('combined_factor_table_path'))
        self.preprocessing_container.add_input(self.combined_factor_table_path)

        self.potential_conversion_mask_path = inputs.File(
            **_create_input_kwargs_from_args_spec('potential_conversion_mask_path'))
        self.preprocessing_container.add_input(self.potential_conversion_mask_path)


        self.spatial_decision_unit_shape = inputs.Text(
            **_create_input_kwargs_from_args_spec('spatial_decision_unit_shape'))
        self.preprocessing_container.add_input(self.spatial_decision_unit_shape)

        self.spatial_decision_unit_area = inputs.Text(
            **_create_input_kwargs_from_args_spec('spatial_decision_unit_area'))
        self.preprocessing_container.add_input(self.spatial_decision_unit_area)

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

        self.frontier_type = inputs.Text(
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
            args[self.marginal_raster_table_path.args_key] = self.marginal_raster_table_path.value()
            args[self.serviceshed_shapefiles_table.args_key] = self.serviceshed_shapefiles_table.value()
            args[self.combined_factor_table_path.args_key] = self.combined_factor_table_path.value()
            args[self.potential_conversion_mask_path.args_key] = self.potential_conversion_mask_path.value()
            args[self.spatial_decision_unit_shape.args_key] = self.spatial_decision_unit_shape.value()
            args[self.spatial_decision_unit_area.args_key] = self.spatial_decision_unit_area.value()

        if self.optimization_container.value():
            args[self.do_optimization.args_key] = self.do_optimization.value()
            args[self.frontier_type.args_key] = self.frontier_type.value()
            args[self.number_of_frontier_points.args_key] = self.number_of_frontier_points.value()
            args[self.objectives_table_path.args_key] = self.objectives_table_path.value()
            args[self.targets_table_path.args_key] = self.targets_table_path.value()

        return args


if __name__ == '__main__':
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
