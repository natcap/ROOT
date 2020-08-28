import os
import sys
import glob
import json
import copy
import datetime
import pprint

import pandas as pd
import numpy as np

from . import integer_solver as isolve
from . import oooe as oooe


def execute(config_file_or_dict):
    """
    Given a config json file or dict, completes the entire optimization processing flow.
    config_file_or_dict must contain:
        - data_folder: folder containing the value tables
        - sdu_id_col: the column in the value tables that indexes each SDU
        - baseline_file (optional): management option to treat as choice 0
        - optimization_folder: output folder
        - analysis_type: one of the options provided by oooe

    config_file_or_dict must also contain values that describe the optimization
    problem to be solved. These will depend on the analysis type.

    :param config_file_or_dict:
    :return:
    """

    config = ensure_args_dict(config_file_or_dict)
    print('config')
    pprint.pprint(config)

    analysis_def = make_analysis_def(config)
    print('analysis_def')
    pprint.pprint(analysis_def)

    print('assembling optimizer data')
    data_for_optimizer = oooe.Data(config['data_folder'],
                                   config['sdu_id_col'],
                                   data_cols=get_data_cols(config),
                                   baseline_file=os.path.join(config['data_folder'],
                                                              config['baseline_file']))

    print('beginning optimization run(s)')
    analysis = make_analysis_object(data_for_optimizer, analysis_def)
    analysis.run()

    if not os.path.exists(config['optimization_folder']):
        os.makedirs(config['optimization_folder'])
    analysis.save_summary_table(config['optimization_folder'])
    analysis.save_decision_tables(config['optimization_folder'])
    analysis.save_summed_choices(config['optimization_folder'])

    return analysis


def make_analysis_def(config):
    """
    This function just makes a copy and returns it. It used to do a lot more, but
    that functionality has been moved to the analysis objects.

    :param config:
    :return:
    """
    analysis = copy.deepcopy(config)
    return analysis


def get_data_cols(config):
    """
    extracts the data table column names that will be used by the optimizer.
    these will be the names of the objective and constraint variables.
    """

    if 'weights' in config and isinstance(config['weights'], dict):
        # handles all cases with weights given directly
        objective_cols = list(config['weights'].keys())
    elif 'weights' in config and (isinstance(config['weights'], str) or isinstance(config['weights'], unicode)):
        # handle the weight table option
        objective_cols = get_header_list(config['weights'])
    else:
        objective_cols = []

    if 'constraints' in config:
        constraint_cols = [c[0] for c in config['constraints']]
    else:
        constraint_cols = []

    if 'targets' in config:
        target_cols = list(config['targets'].keys())
    else:
        target_cols = []

    opt_cols = objective_cols + constraint_cols + target_cols

    return opt_cols


def make_analysis_object(data, analysis_def):
    """
    Creates the appropriate oooe.Analysis object.

    :param data:
    :param analysis_def:
    :return:
    """
    atype = analysis_def['analysis_type']
    aclass = oooe.atype_to_class[atype]
    analysis = aclass(data, analysis_def)
    return analysis


def ensure_args_dict(args):
    """
    args can be a dict or a json filename, returns a dict

    :param args:
    :return:
    """
    if isinstance(args, dict):
        return args
    else:
        return json.load(open(args))


def get_header_list(filename, delimiter=','):
    """
    Reads the first line of filename, splits it with delimiter, returns the
    resulting list.

    :param filename:
    :param delimiter:
    :return:
    """
    with open(filename, 'rU') as f:
        header_str = f.readline().strip()
        header_list = header_str.split(delimiter)
    return header_list


"""
OUTDATED
"""


def call_optimizer(data, analysis_type):
    if analysis_type['analysis_type'] == 'single':
        results = isolve.single_point(data, analysis_type)
    elif analysis_type['analysis_type'] == 'frontier':
        analysis_type['npts'] = int(np.ceil(analysis_type['npts']/2))
        if analysis_type['use_mp'] is True:
            results = isolve.mp_cross_spacing(data, analysis_type)
        else:
            results = isolve.make_frontier_cross_spacing(data, analysis_type)
    elif analysis_type['analysis_type'] == 'n_dim_frontier':
        results = isolve.make_n_dim_frontier(data, analysis_type)
    elif analysis_type['analysis_type'] == 'n_dim_outline':
        results = isolve.make_n_dim_outline(data, analysis_type)
    else:
        print('invalid analysis_type: {}').format(analysis_type['analysis_type'])
        sys.exit(0)

    return results


def construct_optim_data(data_dir, service_list, sdu_id_col,
    baseline_file=None, file_filter=None):
    """
    This function will grab all the .csv files in data_dir, load them as pandas
    data frames, concatenate then construct single service tables.

    The return tables have the structure:
    HRU  LULC1  LULC2  LULC3 ....
     id    x      x      x

    where the xs are the value of the given service under each land use option.

    If a baseline_file argument is given, it will ensure that baseline is in
    position 0. Otherwise the files will be loaded alphabetically.

    args:

    file_filter: list of files to be included. If this is None, then all .csvs
        will be loaded. If not, only the named files will be
    """

    if file_filter is None:
        data_files = sorted(glob.glob(os.path.join(data_dir,'*.csv')))
    else:
        data_files = sorted([os.path.join(data_dir, f) for f in file_filter])

    if baseline_file is not None and baseline_file in data_files:
        data_files.remove(baseline_file)
        data_files.insert(0,baseline_file)

    dfs = [pd.read_csv(f) for f in data_files]
    for df in dfs:
        df.sort_values(by=sdu_id_col, inplace=True)

    opt_tables = dict()
    opt_tables['factornames'] = []

    matshape = (len(dfs[0]), len(data_files))
    for s in service_list:
        opt_tables['factornames'].append(s)
        mat = np.zeros(matshape)
        for i, df in enumerate(dfs):
            mat[:,i] = np.array(df[s])
            mat[np.isnan(mat)] = 0
        opt_tables[s] = mat

    return opt_tables


def construct_optimizer_solution_table(sol, data_dir, sdu_id_col,
        baseline_file=None, file_filter=None, point_num=None):
    '''
    This function creates a pandas dataframe for a given optimizer solution.
    For each SDU, it pulls the row from the appropriate input data file.
    '''

    if file_filter is None:
        data_files = sorted(glob.glob(os.path.join(data_dir,'*.csv')))
    else:
        data_files = sorted([os.path.join(data_dir, f) for f in file_filter])

    if baseline_file is not None and baseline_file in data_files:
        data_files.remove(baseline_file)
        data_files.insert(0,baseline_file)

    # print(data_files)
    dfs = [pd.read_csv(f) for f in data_files]
    for i, df in enumerate(dfs):
        df.sort_values(by=sdu_id_col, inplace=True)
        # print(len(df))
        # print(sol[:,i].shape)
        df['data_array_index'] = i
        df['solution_value'] = sol[:,i]

    data_df = pd.concat(dfs)

    # ECOS does not quite achieve the numerical precision of glpk, so we round off
    # the solutions that are essentially equal to 0 or 1.
    for val in (0.0, 1.0):
        x = data_df['solution_value']
        x[np.isclose(data_df['solution_value'], val, atol=1.e-5)] = val
        data_df['solution_value'] = x

    return data_df


"""
    CODE CUT FROM execute()

    # construct solution tables and save to output dir
    if analysis_def['analysis_type'] == 'single':
        df = construct_optimizer_solution_table(results[0]['sol'],
                data_folder, config['sdu_id_col'], file_filter=file_filter)
        # save solution tables to output dir
        output_file = os.path.join(config['optimization_folder'],
                                   config['result_table_name'])
        df.to_csv(output_file, index=False)

    elif analysis_def['analysis_type'] in ['frontier', 'n_dim_frontier', 'n_dim_outline']:
        for i in range(len(results)):
            df = construct_optimizer_solution_table(results[i]['sol'],
                    data_folder, config['sdu_id_col'], file_filter=file_filter)
            df = df[df['data_array_index'] == 1]
            # save solution tables to output dir
            output_file = os.path.join(config['optimization_folder'],
                                       config['result_table_name'][:-4]+str(i) + '.csv')
            df.to_csv(output_file, index=False)

    # make solution summary table
    if analysis_def['analysis_type'] in ['frontier', 'n_dim_frontier',
                                       'n_dim_outline']:
        summary_table_path = os.path.join(config['optimization_folder'],
                                          'summary_table.csv')
        solution_files = glob.glob(
            os.path.join(config['optimization_folder'], 'optimization_solution*.csv'))
        print(solution_files)
        sol_dfs = [pd.read_csv(sf) for sf in solution_files]
        sum_df = pd.DataFrame()
        sol_columns = sol_dfs[0].columns

        for col_name in sol_columns:
            sum_df[col_name] = [sum(sdf[col_name] * sdf['solution_value']) for
                                sdf in sol_dfs]

        sum_df['pt_id'] = range(len(sol_dfs))

        if analysis_def['analysis_type'] == 'n_dim_outline':
            # in this case we want to add an edge id column
            sum_df['edge_id'] = [r['edge_id'] for r in results]

        sum_df.to_csv(summary_table_path, index=False)

    # create weights table for the monte carlo frontier method
    if analysis_def['analysis_type'] == 'n_dim_frontier':
        obj_keys = sorted(results[0]['weights'].keys())
        weight_table_data = {}
        for ok in obj_keys:
            v = [results[i]['weights'][ok] for i in range(len(results))]
            weight_table_data[ok] = v
        df = pd.DataFrame(data=weight_table_data)
        df['pt_id'] = range(len(results))
        df.to_csv(os.path.join(config['optimization_folder'], 'weight_table.csv'), index=False)
"""

"""
The config vars are:

data_folder: path to a folder with csvs. Each csv represents the value to
each decision unit for a given choice.
baseline_file: this is the default choice in the optimizer (necessary?!?)

example for a frontier analysis:
{
  "cell_size": 3000.0,
  "sdu_id_col": "SDU_ID",
  "data_folder": "/Users/hawt0010/Documents/root_workspace/test_rework/marginal_tables",
  "baseline_file": "baseline.csv",
  "targettypes": {
    "maskpixha": "="
  },
  "flatten_objectives": false,
  "optimization_objectives": {
    "sed_hyd": -1.0,
    "phos_wetl": -1.0
  },
  "output_folder": "/Users/hawt0010/Documents/root_workspace/test_rework",
  "csv_filepath": "/Users/hawt0010/Documents/root_workspace/test_rework/marginal_tables/marginals.csv",
  "optimization_folder": "/Users/hawt0010/Documents/root_workspace/test_rework/optimizations",
  "result_table_name": "optimization_solutions.csv",
  "workspace": "/Users/hawt0010/Documents/root_workspace/test_rework",
  "use_linear_vars": true,
  "normalize_objectives": true,
  "npts": 15,
  "analysis_type": "frontier",
  "targets": {
    "maskpixha": 20000.0
  },
  "grid_type": "hexagon"
}
for an individual optimization, the data and problem defs are:
data should be a dict structured as follows:
    'factornames': [ names of factors ]
    'factorname1': data for factor 1
    'factorname2': data for factor 2
    ...
    'choicenames': [name for each discretechoice] (optional)

problem should be a dict structured as follows:
    'weights': {factornames}: weight for each factor <- i.e. another dict
    'targets': {factornames}: target for each factor with a target
    'targettypes': {factornames}: '<', '>', or '=' (inequals will allow ==)
    'fixedvals': {index: value}: x_index will be constrained to value
    'excludedvals' {index: list}: x_index_j will be set to zero for each
                                  j in list

"""

"""
example config:
{
    'analysis_type': u'n_dim_frontier',
    'baseline_file': 'baseline.csv',
    'data_folder': u'/Users/hawt0010/Documents/root_workspace/test_rework/marginal_tables',
    'flatten_objectives': False,
    'normalize_objectives': True,
    'npts': 15,
    'optimization_folder': u'/Users/hawt0010/Documents/root_workspace/test_rework/optimizations',
    'optimization_objectives': {'n_export': -1.0,
                                'phos_wetl': -1.0,
                                'sed_hyd': -1.0},
    'result_table_name': 'optimization_solutions.csv',
    'sdu_id_col': 'SDU_ID',
    'targets': {'maskpixha': 20000.0},
    'targettypes': {'maskpixha': '='},
    'use_linear_vars': True,
    'workspace': u'/Users/hawt0010/Documents/root_workspace/test_rework'
}

"""

"""
example analysis_def:
{
    'analysis_type': u'n_dim_frontier',
    'flatten_objectives': False,
    'normalize_objectives': True,
    'npts': 15,
    'objectives': ['sed_hyd', 'phos_wetl', 'n_export'],
    'targets': {'maskpixha': 20000.0},
    'targettypes': {'maskpixha': '='},
    'use_mp': False,
    'weights': {'n_export': -1.0, 'phos_wetl': -1.0, 'sed_hyd': -1.0}
}
"""
