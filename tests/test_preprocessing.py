import os
from re import M
import pytest
import json

from natcap.root import rootcore
from natcap.root import preprocessing


test_data_root = "tests/test_data/sample_data"


def test_run_preprocessing():
    args_file = "tests/test_data/correct_ui_args_example.json"
    ui_args = json.load(open(args_file, "r"))["args"]
    internal_args = rootcore.parse_args(ui_args)
    preprocessing.execute(internal_args)


def test_create_overlapping_activity_mask():
    """
    Tests the activity mask union function. Right now just makes sure it runs, doesn't
    do any check for correctness.
    """
    cropland_mask_file = "tests/test_data/sample_data/Ghana/cropland_mask.tif"
    forestry_mask_file = "tests/test_data/sample_data/Ghana/forestry_mask.tif"
    test_workspace = "tests/test_workspace/test_create_overlapping_activity_mask"
    if not os.path.isdir(test_workspace):
        os.makedirs(test_workspace)
    
    mask_path_list = [cropland_mask_file]
    target_file = os.path.join(test_workspace, "single_mask.tif")
    preprocessing._create_overlapping_activity_mask(
        mask_path_list,
        target_file
    )

    mask_path_list = [cropland_mask_file, forestry_mask_file]
    target_file = os.path.join(test_workspace, "double_mask.tif")
    preprocessing._create_overlapping_activity_mask(
        mask_path_list,
        target_file
    )


def test_create_value_tables_for_activity_case_1():
    """
    Test for the rewritten aggregation function.

    Case 1 is the simple case where we don't have mask or a baseline "fill" raster
    """

    sdu_raster_path = os.path.join(test_data_root, "sdu", "sdu_grid.tif")
    sdu_grid_path = os.path.join(test_data_root, "sdu", "sdu_grid.shp")
    vr_folder = os.path.join(test_data_root, "impact_potential_rasters")
    value_raster_lookup = {
        "carbon": os.path.join(vr_folder, "mv_carbon_ag_bmps.tif"),
        "ndr": os.path.join(vr_folder, "mv_ndr_ag_bmps.tif"),
        "sdr": os.path.join(vr_folder, "mv_sdr_ag_bmps.tif"),
        "swy": os.path.join(vr_folder, "mv_swy_ag_bmps.tif")
    }
    activity_list = ["ag_bmps", "forest_restoration", "riparian_restoration"]
    activity_name = "ag_bmps"
    target_folder = "tests/test_workspace/test_create_value_tables_for_activity_case_1"
    if not os.path.isdir(target_folder):
        os.makedirs(target_folder)


    preprocessing._create_value_tables_for_activity(
        sdu_raster_path,
        sdu_grid_path,
        value_raster_lookup,
        activity_list,
        activity_name,
        target_folder
    )