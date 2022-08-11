"""ROOT InVEST Model."""

import os
import sys
import logging
import csv
import multiprocessing
from math import sqrt

import pandas as pd
from osgeo import ogr
from osgeo import gdal
import pygeoprocessing

from natcap.root import __version__
from natcap.root import preprocessing
from natcap.root import postprocessing
from natcap.root import optimization
from natcap.root import arith_parser as ap

LOGGER = logging.getLogger(__name__)


class RootInputError(Exception):
    pass


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

    raster_table = None  # will be overwritten in "do_preprocessing" step but stay None if that is skipped

    if ui_args['do_preprocessing']:

        validate_activity_mask_table(ui_args['activity_mask_table_path'])
        validate_raster_input_table(ui_args['impact_raster_table_path'])
        validate_activity_names_in_amt_and_iprt(ui_args['activity_mask_table_path'],
                                                ui_args['impact_raster_table_path'])
        validate_shapefile_input_table(ui_args['serviceshed_shapefiles_table'])
        validate_cft_table(ui_args['impact_raster_table_path'],
                           ui_args['serviceshed_shapefiles_table'],
                           ui_args['combined_factor_table_path'])
        validate_sdu_shape_arg(ui_args['spatial_decision_unit_shape'])

        root_args['activity_mask_table_path'] = ui_args['activity_mask_table_path']
        root_args['grid_type'] = ui_args['spatial_decision_unit_shape']
        cell_area = float(ui_args['spatial_decision_unit_area']) * 10000
        if root_args['grid_type'] == 'square':
            root_args['cell_size'] = sqrt(cell_area)
        elif root_args['grid_type'] == 'hexagon':
            a = sqrt((2*cell_area) / (3*sqrt(3)))
            root_args['cell_size'] = 2 * a

        root_args['csv_output_folder'] = os.path.join(root_args['workspace'], 'sdu_value_tables')

        root_args['activity_masks'] = _process_activity_mask_table(
            ui_args['activity_mask_table_path'])
        root_args['raster_table'] = _process_raster_table(ui_args['impact_raster_table_path'])
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
            ui_args['workspace_dir'],
            raster_table,
            root_args["combined_factors"]
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


def _process_activity_mask_table(amtpath):
    """
    Reads the activity mask table into a dictionary. The expected format of the
    table is cols 'activity', 'mask_path', and corresponding rows.
    The created dictionary will have the activities as keys and paths as values.
    :param amtpath:
    :return:
    """
    amtdict = {}
    with open(amtpath, 'r') as f:
        f.readline()  # discard header
        for row in f:
            row_fields = [f.strip() for f in row.split(',')]
            amtdict[row_fields[0]] = row_fields[1]
    return amtdict


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


def validate_activity_mask_table(activity_mask_table_path):
    """Check for column names and file existence.

    Expect a table with columns 'activity' and 'mask_path'. Entries in 'mask_path' must point
    to existing files. This function only confirms existence, not filetype.

    Args:
        activity_mask_table_path:

    Returns:
        None
    """
    amt = pd.read_csv(activity_mask_table_path)

    amt_req_cols = ['activity', 'mask_path']
    msg = "Error in AM table: requires two columns named activity and mask_path"
    if len(amt.columns) != len(amt_req_cols):
        raise RootInputError(msg)
    for c, rc in zip(amt.columns, amt_req_cols):
        if c == rc:
            continue
        else:
            raise RootInputError(msg)

    not_found = []
    for activity, fp in zip(amt['activity'], amt['mask_path']):
        if not os.path.isfile(fp):
            not_found.append((activity, fp))
    if len(not_found) > 0:
        msg = "Error in AM table: the following mask rasters could not be located. Please check the filepaths:\n"
        for missing in not_found:
            msg += f"\t{missing[0]}: {missing[1]}\n"
        raise RootInputError(msg)


def validate_raster_input_table(raster_table_path):
    """
    Check to make sure that all the rasters named in `raster_table_path` exist.

    Skips first column of the table (assumes it is a label column, not a file path column).
    """
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


def validate_activity_names_in_amt_and_iprt(amt_path, iprt_path):
    """
    Checks to make sure that the same activity names are use in the activity mask table and impact
    potential raster table. Each activity named in the IPRT should have a mask, and vice versa.

    Args:
        amt_path:
        iprt_path:

    Raises:
        RootInputError

    Returns:
        None
    """
    amt = pd.read_csv(amt_path)
    iprt = pd.read_csv(iprt_path)

    amt_activities = list(amt['activity'])
    iprt_activities = list(iprt.columns[1:])

    match = True
    for a in amt_activities:
        if a not in iprt_activities:
            match = False
    for a in iprt_activities:
        if a not in amt_activities:
            match = False
    if match is False:
        msg = "Error: activities do not match in Activity Mask table and Impact Potential Raster table. Please check spelling and missing/extra values: {} vs {}".format(amt_activities, iprt_activities)
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
    """

    Args:
        rt_path:
        st_path:
        cft_path:

    Returns:

    """
    if len(cft_path) == 0:
        return
    cft = pd.read_csv(cft_path)

    # check for correct headers
    missing_name = False
    missing_factors = False
    if 'name' not in cft.columns:
        missing_name = True
    if 'factors' not in cft.columns:
        missing_factors = True
    if missing_name or missing_factors:
        raise RootInputError("CFT table is missing columns 'name' and/or 'factors'")

    # get factors listed in raster table
    rt = pd.read_csv(rt_path)
    raster_factors = list(rt['name'])
    activity_area_factors = ['{}_ha'.format(f) for f in rt.columns]

    # get factors listed in shapefile table (if provided - shapefile table is optional)
    shape_factors = []
    if len(st_path) > 0:
        st = pd.read_csv(st_path)
        for _, row in st.iterrows():
            sname = row['name']
            shape_factors.append(sname)
            wcols = row['weight_col'].split(' ')
            for wc in wcols:
                shape_factors.append('{}_{}'.format(sname, wc))

    # combined factors - also include math names as valid
    all_factors = raster_factors + shape_factors + activity_area_factors + ap.ALL_AP_TOKENS

    invalid_factors = []
    for _, row in cft.iterrows():
        name = row['name']
        factors = ap._tokenize(row['factors'])
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


def validate_objectives_and_constraints_tables(obj_table_file, cons_table_file, workspace, raster_table, combined_factors):

    if raster_table is not None:
        # the case where we ran the preprocessing step
        factors = raster_table.factor_names + list(combined_factors.keys()) + \
            [f"{a}_ha" for a in raster_table.activity_names]
        print(f"factors: {factors}")
    else:
        # this is the case if we are skipping the preprocessing step, which means that we
        # should already have generated the required files
        baseline_sdu_stats_file = os.path.join(workspace, 'sdu_value_tables', 'baseline.csv')
        if not os.path.isfile(baseline_sdu_stats_file):
            msg = f"Missing {baseline_sdu_stats_file}. Do you need to run preprocessing?"
            raise RootInputError(msg)
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
            c_expr = row_vals[0]
            c_factors = ap._tokenize(c_expr)
            for f in c_factors:
                if f not in factors and f not in ap.ALL_AP_TOKENS:
                    try:
                        float(f)
                        continue
                    except ValueError:
                        not_found.append(f)

        if len(not_found) > 0:
            msg = "Error in Targets table. The following factors were not found in the sdu value tables: {}".format(
                not_found
            )
            raise RootInputError(msg)



if __name__ == '__main__':
    multiprocessing.freeze_support()

    if '--test-imports' in sys.argv:
        sys.exit(0)

    if '-v' in sys.argv:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
