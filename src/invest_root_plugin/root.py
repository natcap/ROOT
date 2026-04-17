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
from natcap.invest import spec
from natcap.invest import validation
import pygeoprocessing

# from . import __version__
from .model_spec import MODEL_SPEC
from . import rootcore
from . import preprocessing
from . import postprocessing
from . import optimization
from . import arith_parser as ap

LOGGER = logging.getLogger(__name__)


class RootInputError(Exception):
    pass




def execute(args):
    """root.

    """
    # LOGGER.info(f'Running ROOT version {__version__}')
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


@validation.invest_validator
def validate(args):
    return validation.validate(args, MODEL_SPEC)

    # validation_warnings = validation.validate(
    #     args, ARGS_SPEC['args'], ARGS_SPEC['args_with_spatial_overlap'])

    # invalid_keys = validation.get_invalid_keys(validation_warnings)

    # if 'spatial_decision_unit_shape' not in invalid_keys:
    #     sdu_valid = True
    #     try:
    #         if not validate_sdu_shape_arg(args['spatial_decision_unit_shape']):
    #             sdu_valid = False
    #     except RootInputError:
    #         sdu_valid = False

    #     if not sdu_valid:
    #         validation_warnings.append(
    #             (['spatial_decision_unit_shape'],
    #              ('Spatial Decision Unit Shape must be "square", "hexagon", '
    #               'or a path to a vector')))

    # return validation_warnings


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


# class Root(model.InVESTModel):
#     def __init__(self):
#         model.InVESTModel.__init__(
#             self,
#             label='ROOT',
#             target=execute,
#             validator=validate,
#             localdoc=u'../documentation/root.html'
#         )

#         # Explicitly setting the default workspace because the InVEST UI's
#         # approach relies on self.target.__module__, which isn't reliable when
#         # execute is in the same script as the launcher.  In this case, the
#         # module name is __main__.  Technically true, but not user-readable.
#         self.workspace.set_value(os.path.normpath(
#             os.path.expanduser('~/Documents/root_workspace')))

#         self.preprocessing_container = inputs.Container(
#             args_key=u'preprocessing_container',
#             expandable=False,
#             expanded=True,
#             label=u'Preprocessing Arguments')
#         self.add_input(self.preprocessing_container)

#         self.do_preprocessing = inputs.Checkbox(
#             **_create_input_kwargs_from_args_spec(
#                 'do_preprocessing', validate=False))
#         self.preprocessing_container.add_input(self.do_preprocessing)

#         self.activity_mask_raster_path = inputs.File(
#             **_create_input_kwargs_from_args_spec('activity_mask_table_path'))
#         self.preprocessing_container.add_input(self.activity_mask_raster_path)

#         self.impact_raster_table_path = inputs.File(
#             **_create_input_kwargs_from_args_spec('impact_raster_table_path'))
#         self.preprocessing_container.add_input(self.impact_raster_table_path)

#         self.serviceshed_shapefiles_table = inputs.File(
#             **_create_input_kwargs_from_args_spec('serviceshed_shapefiles_table'))
#         self.preprocessing_container.add_input(self.serviceshed_shapefiles_table)

#         self.combined_factor_table_path = inputs.File(
#             **_create_input_kwargs_from_args_spec('combined_factor_table_path'))
#         self.preprocessing_container.add_input(self.combined_factor_table_path)

#         self.spatial_decision_unit_shape = inputs.Text(
#             **_create_input_kwargs_from_args_spec('spatial_decision_unit_shape'))
#         self.preprocessing_container.add_input(self.spatial_decision_unit_shape)

#         self.spatial_decision_unit_area = inputs.Text(
#             **_create_input_kwargs_from_args_spec('spatial_decision_unit_area'))
#         self.preprocessing_container.add_input(self.spatial_decision_unit_area)

#         self.aoi_file_path = inputs.Text(
#             **_create_input_kwargs_from_args_spec('aoi_file_path'))
#         self.preprocessing_container.add_input(self.aoi_file_path)

#         self.advanced_args_json_path = inputs.Text(
#             **_create_input_kwargs_from_args_spec('advanced_args_json_path'))
#         self.preprocessing_container.add_input(self.advanced_args_json_path)


#         self.optimization_container = inputs.Container(
#             args_key=u'optimization_container',
#             expandable=False,
#             expanded=True,
#             label=u'Optimization Arguments')
#         self.add_input(self.optimization_container)
#         self.do_optimization = inputs.Checkbox(
#             **_create_input_kwargs_from_args_spec(
#                 'do_optimization', validate=False))
#         self.optimization_container.add_input(self.do_optimization)

#         self.optimization_suffix = inputs.Text(
#             **_create_input_kwargs_from_args_spec('optimization_suffix'))
#         self.optimization_container.add_input(self.optimization_suffix)

#         self.frontier_type = inputs.Text(
#             **_create_input_kwargs_from_args_spec('frontier_type'))
#         self.optimization_container.add_input(self.frontier_type)

#         self.number_of_frontier_points = inputs.Text(
#             **_create_input_kwargs_from_args_spec('number_of_frontier_points'))
#         self.optimization_container.add_input(self.number_of_frontier_points)

#         self.objectives_table_path = inputs.File(
#             **_create_input_kwargs_from_args_spec('objectives_table_path'))
#         self.optimization_container.add_input(self.objectives_table_path)

#         self.targets_table_path = inputs.File(
#             **_create_input_kwargs_from_args_spec('targets_table_path'))
#         self.optimization_container.add_input(self.targets_table_path)

#     def assemble_args(self):
#         args = {
#             self.workspace.args_key: self.workspace.value(),
#             self.suffix.args_key: self.suffix.value(),
#             self.preprocessing_container.args_key: self.preprocessing_container.value(),
#             self.optimization_container.args_key: self.optimization_container.value(),
#         }
#         if self.optimization_suffix.value():
#             args[self.optimization_suffix.args_key] = self.optimization_suffix.value()
#         if self.preprocessing_container.value():
#             args[self.do_preprocessing.args_key] = self.do_preprocessing.value()
#             args[self.activity_mask_raster_path.args_key] = self.activity_mask_raster_path.value()
#             args[self.impact_raster_table_path.args_key] = self.impact_raster_table_path.value()
#             args[self.serviceshed_shapefiles_table.args_key] = self.serviceshed_shapefiles_table.value()
#             args[self.combined_factor_table_path.args_key] = self.combined_factor_table_path.value()
#             args[self.spatial_decision_unit_shape.args_key] = self.spatial_decision_unit_shape.value()
#             args[self.spatial_decision_unit_area.args_key] = self.spatial_decision_unit_area.value()
#             args[self.aoi_file_path.args_key] = self.aoi_file_path.value()
#             args[self.advanced_args_json_path.args_key] = self.advanced_args_json_path.value()

#         if self.optimization_container.value():
#             args[self.do_optimization.args_key] = self.do_optimization.value()
#             args[self.frontier_type.args_key] = self.frontier_type.value()
#             args[self.number_of_frontier_points.args_key] = self.number_of_frontier_points.value()
#             args[self.objectives_table_path.args_key] = self.objectives_table_path.value()
#             args[self.targets_table_path.args_key] = self.targets_table_path.value()

#         return args


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
