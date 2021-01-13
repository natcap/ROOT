""""
This is a saved model run from root.
Generated: Fri Apr 20 14:49:39 2018
InVEST version: 3.3.3
"""

from natcap.root import root


args = {
        u'activity_mask_table_path': u'/Users/hawt0010/Projects/ROOT/training/Medellin_data_ROOT_training/Training_Data/Tables/0_am_table.csv',
        u'combined_factor_table_path': u'/Users/hawt0010/Projects/ROOT/training/Medellin_data_ROOT_training/Training_Data/Tables/3_cf_table.csv',
        u'do_optimization': True,
        u'do_preprocessing': False,
        u'frontier_type': u'n_dim_frontier',
        u'marginal_raster_table_path': u'/Users/hawt0010/Projects/ROOT/training/Medellin_data_ROOT_training/Training_Data/Tables/1_ipm_table.csv',
        u'number_of_frontier_points': 20.0,
        u'objectives_table_path': u'/Users/hawt0010/Projects/ROOT/training/Medellin_data_ROOT_training/Training_Data/Tables/4_objectives_table.csv',
        u'optimization_container': False,
        u'preprocessing_container': False,
        u'serviceshed_shapefiles_table': u'',
        u'spatial_decision_unit_area': 100.0,
        u'spatial_decision_unit_shape': u'hexagon',
        u'targets_table_path': u'/Users/hawt0010/Projects/ROOT/training/Medellin_data_ROOT_training/Training_Data/Tables/5_constraints_table.csv',
        u'workspace': u'/Users/hawt0010/Projects/ROOT/training/Medellin_data_ROOT_training/Training_Data/Workspace_arithCFT',
}

if __name__ == '__main__':
    root.execute(args)
