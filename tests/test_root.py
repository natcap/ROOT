import pytest
import json

from natcap.root import rootcore


# def list_elements_equal(list_a, list_b):
#     return sorted(list_a) == sorted(list_b)

@pytest.fixture(scope='module')
def raster_table():
    rt = rootcore._process_raster_table('tests/test_data/test_raster_table.csv')
    return rt


def test_parse_args():
    args_file = "tests/test_data/correct_ui_args_example.json"
    ui_args = json.load(open(args_file, "r"))["args"]
    internal_args = rootcore.parse_args(ui_args)


def test_process_raster_table(raster_table):
    assert sorted(raster_table.activity_names) == ['act1', 'act2', 'act3']
    assert sorted(raster_table.factor_names) == ['fact1', 'fact2', 'fact3']
    assert raster_table['act2']['fact3'] == 'F3A2'


def test_minmax_objectives_table():
    # TODO this test uses the old style of weights table - need to update when I update the table format
    ui_args = {'objectives_table_path': 'tests/test_data/test_objective_table_minmax.csv'}
    root_args = {'analysis_type': 'n_dim_frontier'}
    opt_obj = rootcore._process_objectives_table(ui_args, root_args)
    assert opt_obj['s_export'] == -1.0
    assert opt_obj['n_export'] == -1.0
    assert opt_obj['c_store'] == 1.0

    # for weight table, the UI just passes the table file on so that the Analysis
    # object can handle it
    root_args['analysis_type'] = 'weight_table'
    opt_obj = rootcore._process_objectives_table(ui_args, root_args)
    assert opt_obj == 'tests/test_data/test_objective_table_minmax.csv'


def test_constraints_table():
    ui_args = {
        'targets_table_path': 'tests/test_data/test_targets_table.csv'
    }
    constraints = rootcore._process_constraints_table(ui_args)

    for v, w in zip(constraints[0], ['maskpixha', '=', 20000]):
        assert v == w
    for v, w in zip(constraints[1], ['cost', '<=', 1000000]):
        assert v == w
