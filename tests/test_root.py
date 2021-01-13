import pytest

from natcap.root import root


def list_elements_equal(list_a, list_b):
    return cmp(list_a, list_b) == 0


@pytest.fixture(scope='module')
def raster_table():
    rt = root._process_raster_table('/Users/hawt0010/Projects/ROOT/root_source/test_data/test_raster_table.csv')
    return rt


def test_process_raster_table(raster_table):
    assert list_elements_equal(raster_table.activity_names, ['act1', 'act2', 'act3'])
    assert list_elements_equal(raster_table.factor_names, ['fact1', 'fact2', 'fact3'])
    assert raster_table['act2']['fact3'] == 'F3A2'


def test_minmax_objectives_table():
    # TODO this test uses the old style of weights table - need to update when I update the table format
    ui_args = {'objectives_table_path': '/Users/hawt0010/Projects/ROOT/root_source/test_data/test_objective_table_minmax.csv'}
    root_args = {'analysis_type': 'n_dim_frontier'}
    opt_obj = root._process_objectives_table(ui_args, root_args)
    assert opt_obj['s_export'] == -1.0
    assert opt_obj['n_export'] == -1.0
    assert opt_obj['c_store'] == 1.0

    # for weight table, the UI just passes the table file on so that the Analysis
    # object can handle it
    root_args['analysis_type'] = 'weight_table'
    opt_obj = root._process_objectives_table(ui_args, root_args)
    assert opt_obj == '/Users/hawt0010/Projects/ROOT/root_source/test_data/test_objective_table_minmax.csv'


def test_constraints_table():
    ui_args = {
        'targets_table_uri': '/Users/hawt0010/Projects/ROOT/root_source/test_data/test_targets_table.csv'
    }
    constraints = root._process_constraints_table(ui_args)

    for v, w in zip(constraints[0], ['maskpixha', '=', 20000]):
        assert v == w
    for v, w in zip(constraints[1], ['cost', '<=', 1000000]):
        assert v == w
