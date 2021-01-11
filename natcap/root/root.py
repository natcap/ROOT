"""ROOT InVEST Model."""

import os
import sys
import logging
import collections
import csv
import uuid
import json
from math import sqrt

import pandas as pd
from osgeo import ogr
from osgeo import gdal
from natcap.invest.ui import model, inputs

sys.path.extend([os.getcwd()])

from natcap.root import preprocessing
from natcap.root import postprocessing
from natcap.root import optimization

LOGGER = logging.getLogger(__name__)


class RootInputError(Exception):
    pass


def execute(args):
    """root.

    """

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
            with open(ui_args['combined_factor_table_path'], 'rU') as tablefile:
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
            with open(filename, 'rU') as tablefile:
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
        with open(ui_args['objectives_table_path'], 'rU') as tablefile:
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
    required_keys = [
        'marginal_raster_table_path',
        'potential_conversion_mask_path',
        'spatial_decision_unit_shape',
        'spatial_decision_unit_area',
    ]
    if 'optimization_container' in args and args['optimization_container']:
        required_keys.extend([
            'frontier_type',  # ony when optimization_container
            'number_of_frontier_points',  # only when optimization_container
            'objectives_table_path', # only when optimization_container
            'targets_table_path',  # only when optimization_container
        ])

    missing_key_list = []
    no_value_list = []
    validation_error_list = []
    for key in required_keys:
        if limit_to is None or limit_to == key:
            if key not in args:
                missing_key_list.append(key)
            elif args[key] in ('', None):
                no_value_list.append(key)


    if (limit_to in ['potential_conversion_mask_path', None] and
            'potential_conversion_mask_path' in args):
        raster = gdal.OpenEx(args['potential_conversion_mask_path'],
                             gdal.OF_RASTER)
        if raster is None:
            validation_error_list.append(([key], 'Must be a raster'))
        raster = None

    if (limit_to in ['targets_table_path', None] and
            'targets_table_path' in args):
        dataframe = pd.read_csv(args['targets_table_path'])
        required_colnames = set(['name', 'cons_type', 'value'])
        found_colnames = set(list(dataframe))
        missing_colnames = required_colnames - found_colnames
        if len(missing_colnames) > 0:
            validation_error_list.append(
                (['targets_table_path'],
                 'Table is missing required columns: %s' % sorted(
                     missing_colnames)))

    if (limit_to in ['frontier_type', None] and
            'frontier_type' in args):
        if args['frontier_type'] not in (
                'weight_table', 'frontier', 'n_dim_frontier',
                'n_dim_outline'):
            validation_error_list.append(
                (['frontier_type'], 'Invalid frontier type provided'))

    for number_key in ('number_of_frontier_points',
                       'spatial_decision_unit_area'):
        if limit_to in [number_key, None]:
            if number_key in args:
                try:
                    float(args['number_key'])
                except (ValueError, TypeError):
                    # ValueError when empty string
                    # TypeError when None
                    validation_error_list.append(([key], 'Must be a number'))

    return validation_error_list


class Root(model.InVESTModel):
    def __init__(self):
        model.InVESTModel.__init__(
            self,
            label=u'ROOT',
            target=execute,
            validator=validate,
            localdoc=u'../documentation/root.html'
        )

        self.preprocessing_container = inputs.Container(
            args_key=u'preprocessing_container',
            expandable=False,
            expanded=True,
            label=u'Preprocessing Arguments')
        self.add_input(self.preprocessing_container)
        self.do_preprocessing = inputs.Checkbox(
            args_key=u'do_preprocessing',
            helptext=(
                u"Check to create marginal value tables based on "
                u"raster/serviceshed inputs"),
            label=u'Do Preprocessing')
        self.preprocessing_container.add_input(self.do_preprocessing)
        self.marginal_raster_table_path = inputs.File(
            args_key=u'marginal_raster_table_path',
            helptext=(
                u"Table that lists names and filepaths for impact "
                u"potential rasters.<br><br>ROOT will aggregate these "
                u"rasters to the spatial units outlined by the SDU "
                u"grid.  See User's Guide for details on file format."),
            label=u'Impact Potential Raster Table (CSV)',
            validator=self.validator)
        self.preprocessing_container.add_input(self.marginal_raster_table_path)
        self.serviceshed_shapefiles_table = inputs.File(
            args_key=u'serviceshed_shapefiles_table',
            helptext=(
                u"Table that lists names, uris, and service value "
                u"columns for servicesheds.<br><br>ROOT will calculate "
                u"the overlap area and weighted average of each "
                u"serviceshed for each SDU. See User's Guide for "
                u"details on file format."),
            label=u'Spatial Weighting Maps Table (CSV)',
            validator=self.validator)
        self.preprocessing_container.add_input(self.serviceshed_shapefiles_table)
        self.combined_factor_table_path = inputs.File(
            args_key=u'combined_factor_table_path',
            helptext=(
                u"This table allows the user to construct composite "
                u"factors, such as ecosystem services weighted by a "
                u"serviceshed.  The value in the 'name' column will be "
                u"used to identify this composite factor in summary "
                u"tables, output maps, and in configuration tables in "
                u"the optimization section.  The value in the 'factors' "
                u"field should list which marginal values and "
                u"shapefiles to multiply together to construct the "
                u"composite factor.  The entry should be comma "
                u"separated, such as 'sed_export, hydropower_capacity'. "
                u"Use an '_' to identify fields of interest within a "
                u"serviceshed shapefile."),
            label=u'Composite Factor Table (CSV)',
            validator=self.validator)
        self.preprocessing_container.add_input(self.combined_factor_table_path)
        self.potential_conversion_mask_path = inputs.File(
            args_key=u'potential_conversion_mask_path',
            helptext=(
                u"Raster that indicates which pixels should be "
                u"considered as potential activity locations.  Values "
                u"must be 1 for activity locations or NODATA for "
                u"excluded locations."),
            label=u'Activity Mask Raster',
            validator=self.validator)
        self.preprocessing_container.add_input(self.potential_conversion_mask_path)
        self.spatial_decision_unit_shape = inputs.Text(
            args_key=u'spatial_decision_unit_shape',
            helptext=(
                u"Determines the shape of the SDUs used to aggregate "
                u"the impact potential and spatial weighting maps.  Can "
                u"be square or hexagon to have ROOT generate an "
                u"appropriately shaped regular grid, or can be the full "
                u"path to an existing SDU shapefile.  If an existing "
                u"shapefile is used, it must contain a unique ID field "
                u"SDU_ID."),
            label=u'Spatial Decision Unit Shape',
            validator=self.validator)
        self.preprocessing_container.add_input(self.spatial_decision_unit_shape)
        self.spatial_decision_unit_area = inputs.Text(
            args_key=u'spatial_decision_unit_area',
            helptext=(
                u"Area of each grid cell in the constructed grid. "
                u"Measured in hectares (1 ha = 10,000m^2). This is "
                u"ignored if an existing file is used as the SDU map."),
            label=u'Spatial Decision Unit Area (ha)',
            validator=self.validator)
        self.preprocessing_container.add_input(self.spatial_decision_unit_area)
        self.optimization_container = inputs.Container(
            args_key=u'optimization_container',
            expandable=False,
            expanded=True,
            label=u'Optimization Arguments')
        self.add_input(self.optimization_container)
        self.do_optimization = inputs.Checkbox(
            args_key=u'do_optimization',
            helptext=u'Check to perform optimization',
            label=u'Do Optimization')
        self.optimization_container.add_input(self.do_optimization)
        self.optimization_suffix = inputs.Text(
            args_key=u'optimization_suffix',
            helptext=(
                u"This text will be appended to the optimization "
                u"output folder to distinguish separate analyses."),
            label=u'Optimization Results Suffix (Optional)',
            validator=self.validator)
        self.optimization_container.add_input(self.optimization_suffix)
        self.frontier_type = inputs.Text(
            args_key=u'frontier_type',
            helptext=(
                u"Determines the mode of operation for the optimizer. "
                u"Can be: <ul><li>weight_table: use to provide "
                u"particular objective weights for a given number of "
                u"optimization runs.  </li><li>frontier: evenly-spaced "
                u"frontier (requires exactly two objectives in "
                u"objectives table).</li> <li>n_dim_frontier: use for "
                u"more than two objective analysis.  Randomly samples "
                u"points on N dimensional "
                u"frontier.</li><li>n_dim_outline: constructs 2D "
                u"frontiers for all pairs of objectives.</li></ul>"),
            label=u'Analysis Type',
            validator=self.validator)
        self.optimization_container.add_input(self.frontier_type)
        self.number_of_frontier_points = inputs.Text(
            args_key=u'number_of_frontier_points',
            helptext=(
                u"Determines number of points to calculate.  Note that "
                u"for 'frontier' runs, the actual number of points will "
                u"be higher than this number to ensure whole-frontier "
                u"coverage."),
            label=u'Number of Frontier Points',
            validator=self.validator)
        self.optimization_container.add_input(self.number_of_frontier_points)
        self.objectives_table_path = inputs.File(
            args_key=u'objectives_table_path',
            helptext=(
                u"This table identifies which factors from the "
                u"preprocessing results to use as objectives in the "
                u"optimization.  Values in the 'objective' column "
                u"should match names assigned to marginal values, "
                u"servicesheds, or combined factors.  For the 'single' "
                u"optimization option, the values in 'weight' should be "
                u"the appropriate numeric values.  For all others, the "
                u"'weight' column should say 'minimize' or 'maximize' "
                u"according to whether the optimization should prefer "
                u"lower or higher values for that factor, respectively."),
            label=u'Objectives Table (CSV)',
            validator=self.validator)
        self.optimization_container.add_input(self.objectives_table_path)
        self.targets_table_path = inputs.File(
            args_key=u'targets_table_path',
            helptext=(
                u"This table identifies targets (constraints) to apply "
                u"to the optimization."),
            label=u'Targets Table (CSV)',
            validator=self.validator)
        self.optimization_container.add_input(self.targets_table_path)

        # Set interactivity, requirement as input sufficiency changes


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
    if '--test-imports' in sys.argv:
        sys.exit(0)

    ui = Root()
    ui.run()
    inputs.QT_APP.exec_()
