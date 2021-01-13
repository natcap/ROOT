import pytest
import os
import copy

from rootcode import optim_core


@pytest.fixture(scope='module')
def problem_def():
    prob = {
        'analysis_type': 'single',
        'weights': {'N_LOADING_REL': -1.0, 'S_LOADING_REL': -1.0},
        'constraints': [('AG_PROFIT_REL', '>=', -1000000),
                        ('N_LOADING_REL', '<=', -10)],
        'npts': 10,
        'use_linear_vars': True,
        'normalize_objectives': True
    }
    return prob


@pytest.fixture(scope='module')
def data():
    data_folder = '/Users/hawt0010/Projects/ROOT/root_source/sample_data/multi_bmp'
    data = optim_core.Data(data_folder, 'HRU',
                           data_cols=['N_LOADING_REL',
                                      'P_LOADING_REL',
                                      'S_LOADING_REL',
                                      'AG_PROFIT_REL'])
    return data


def test_weight_table_iterator(problem_def, data):
    output_folder = "/Users/hawt0010/Projects/ROOT/root_source/sample_output/wti"
    pdef = copy.deepcopy(problem_def)
    pdef['weights'] = "/Users/hawt0010/Projects/ROOT/root_source/sample_data/multi_bmp_weight_table.csv"
    wti = optim_core.WeightTableIterator(data, pdef)
    wti.run()
    wti.save_summary_table(output_folder)
    assert os.path.isfile(os.path.join(output_folder, 'summary_table.csv'))
    wti.save_summed_choices(output_folder)
    assert os.path.isfile(os.path.join(output_folder, 'summed_choices.csv'))
    wti.save_decision_tables(output_folder)
    for i in range(wti.npts):
        assert os.path.isfile(os.path.join(output_folder, 'solution_{}.csv'.format(i)))
