import os
import pytest
import json

from natcap.root import rootcore
from natcap.root import preprocessing



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
