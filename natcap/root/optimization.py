import os
import glob
import json
import copy
import pprint

from natcap.root import optim_core as opco


def execute(config_file_or_dict):
    """
    Given a config json file or dict, completes the entire optimization processing flow.
    `config_file_or_dict` must contain:

    - data_folder: folder containing the value tables
    - sdu_id_col: the column in the value tables that indexes each SDU
    - baseline_file: management option to treat as choice 0
    - optimization_folder: output folder
    - analysis_type: one of the options provided by oooe

    `config_file_or_dict` must also contain values that describe the optimization
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
    data_for_optimizer = opco.Data(config['data_folder'],
                                   config['sdu_id_col'],
                                   data_cols=get_data_cols(config),
                                   baseline_file=os.path.join(config['data_folder'],
                                                              config['baseline_file']))

    # analysis_def['excluded_vals'] = make_exclude_list(data_for_optimizer)

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


def make_exclude_list(data):
    return None


def get_activity_names(config, include_baseline=True):
    """
    returns the names of the activities with baseline first

    :param config:
    :return:
    """
    activity_files = sorted(glob.glob(os.path.join(config['data_folder'], '*.csv')))
    activity_files = [os.path.basename(f) for f in activity_files]
    if 'baseline_file' in config:
        activity_files.remove(config['baseline_file'])
        if include_baseline:
            activity_files.insert(0, config['baseline_file'])
    activity_names = [os.path.splitext(f)[0] for f in activity_files]
    return activity_names


def get_data_cols(config):
    """
    extracts the data table column names that will be used by the optimizer.
    these will be the names of the objective and constraint variables.
    """

    if 'weights' in config and isinstance(config['weights'], dict):
        # handles all cases with weights given directly
        objective_cols = list(config['weights'].keys())
    elif 'weights' in config and isinstance(config['weights'], str): # or isinstance(config['weights'], unicode)):
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

    activity_names = get_activity_names(config, include_baseline=False)
    activity_area_cols = [act + '_ha' for act in activity_names]

    opt_cols = sorted(list(set(objective_cols + constraint_cols + target_cols + activity_area_cols + ['exclude'])))
    # probably not the best way to remove duplicates, but whatever

    return opt_cols


def make_analysis_object(data, analysis_def):
    """
    Creates the appropriate oooe.Analysis object.

    :param data:
    :param analysis_def:
    :return:
    """
    atype = analysis_def['analysis_type']
    aclass = opco.atype_to_class[atype]
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
