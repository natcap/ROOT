import os
from re import M
import pytest
import json

import pandas as pd
import numpy as np

from natcap.root import rootcore
from natcap.root import preprocessing


test_data_root = "tests/test_data/sample_data"


# def test_run_preprocessing():
#     args_file = "tests/test_data/correct_ui_args_example.json"
#     ui_args = json.load(open(args_file, "r"))["args"]
#     internal_args = rootcore.parse_args(ui_args)
#     preprocessing.execute(internal_args)


def test_create_overlapping_activity_mask():
    """
    Tests the activity mask union function. Right now just makes sure it runs, doesn't
    do any check for correctness.
    """
    cropland_mask_file = "tests/test_data/sample_data/Ghana/ActivityMasks/cropland_mask.tif"
    forestry_mask_file = "tests/test_data/sample_data/Ghana/ActivityMasks/forestry_mask.tif"
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
    target_folder = "tests/test_workspace/test_create_value_tables_for_activity_case_1"
    if not os.path.isdir(target_folder):
        os.makedirs(target_folder)

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

    preprocessing._create_value_tables_for_activity(
        sdu_raster_path,
        value_raster_lookup,
        activity_list,
        activity_name,
        target_folder,
        sdu_grid_path=sdu_grid_path,
        calc_area_for_activity="ag_bmps"
    )


def test_create_value_tables_for_activity_case_2():
    target_folder = "tests/test_workspace/test_create_value_tables_for_activity_case_2"
    if not os.path.isdir(target_folder):
        os.makedirs(target_folder)

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
    mask_raster_path = os.path.join(test_data_root, "activity_mask_rasters", "ag_bmps_mask.tif")

    preprocessing._create_value_tables_for_activity(
        sdu_raster_path,
        value_raster_lookup,
        activity_list,
        activity_name,
        target_folder,
        sdu_grid_path=sdu_grid_path,
        mask_raster_path=mask_raster_path,
        calc_area_for_activity="ag_bmps"
    )


def dont_test_create_value_tables_for_activity_case_3():
    ghana_folder = os.path.join(test_data_root, "Ghana")
    target_folder = "tests/test_workspace/test_create_value_tables_for_activity_case_3"
    if not os.path.isdir(target_folder):
        os.makedirs(target_folder)

    sdu_raster_path = os.path.join(ghana_folder, "sdu_map", "sdu_raster.tif")
    sdu_grid_path = os.path.join(ghana_folder, "sdu_map", "sdu_grid.shp")
    vr_folder = os.path.join(ghana_folder, "ImpactPotentialMaps")
    value_raster_lookup = {
        "biodiversity": os.path.join(vr_folder, "bmps_biodiversity.tif"),
        "carbon": os.path.join(vr_folder, "bmps_carbon.tif"),
        "cropland_value": os.path.join(vr_folder, "bmps_cropland_value.tif")
    }
    activity_list = ["bmps", "intensification", "restoration"]
    activity_name = "bmps"
    mask_raster_path = os.path.join(ghana_folder, "ActivityMasks", "cropland_mask.tif")

    preprocessing._create_value_tables_for_activity(
        sdu_raster_path,
        value_raster_lookup,
        activity_list,
        activity_name,
        target_folder,
        sdu_grid_path=sdu_grid_path,
        mask_raster_path=mask_raster_path,
        calc_area_for_activity="bmps",
        sdu_id_column="SDU_ID"
    )


def test_aggregation_with_tiny_data():
    src_dir = "tests/test_data/dummy_data"
    target_folder = "tests/test_workspace/test_create_value_tables_tiny_data"
    if not os.path.isdir(target_folder):
        os.makedirs(target_folder)

    sdu_raster_path = os.path.join(src_dir, "sdu.tif")
    sdu_list = [1,2,3,4]
    value_raster_lookup = {
        "A": os.path.join(src_dir, "ones.tif")
    }
    activity_list = ["activity_1", "activity_2"]
    preprocessing._create_value_tables_for_activity(
        sdu_raster_path,
        value_raster_lookup,
        activity_list,
        "activity_1",
        target_folder,
        sdu_list=sdu_list,
        calc_area_for_activity="activity_1"
    )

    df = pd.read_csv(os.path.join(target_folder, "activity_1.csv"))
    df.set_index("SDU_ID", inplace=True)
    for i in [1,2,3,4]:
        assert df.loc[i, "A"] == 25
        assert df.loc[i, "activity_1_ha"] == 225

    # Now with an AOI
    aoi_raster_path = os.path.join(src_dir, "aoi.tif")
    preprocessing._create_value_tables_for_activity(
        sdu_raster_path,
        value_raster_lookup,
        activity_list,
        "activity_1",
        target_folder,
        sdu_list=sdu_list,
        aoi_raster_path=aoi_raster_path,
        calc_area_for_activity="activity_1"
    )
    df = pd.read_csv(os.path.join(target_folder, "activity_1.csv"))
    df.set_index("SDU_ID", inplace=True)
    for i in [1,2]:
        assert df.loc[i, "A"] == 20
        assert df.loc[i, "activity_1_ha"] == 180
    for i in [3,4]:
        assert df.loc[i, "A"] == 25
        assert df.loc[i, "activity_1_ha"] == 225

    # Now with an AOI and a mask
    mask_raster_path = os.path.join(src_dir, "mask.tif")
    preprocessing._create_value_tables_for_activity(
        sdu_raster_path,
        value_raster_lookup,
        activity_list,
        "activity_1",
        target_folder,
        sdu_list=sdu_list,
        aoi_raster_path=aoi_raster_path,
        mask_raster_path=mask_raster_path,
        calc_area_for_activity="activity_1"
    )
    df = pd.read_csv(os.path.join(target_folder, "activity_1.csv"))
    df.set_index("SDU_ID", inplace=True)
    vals_a = [12, 8, 15, 10]
    areas = [108, 72, 135, 90]
    for i in range(len(vals_a)):
        assert df.loc[i+1, "A"] == vals_a[i]
        assert df.loc[i+1, "activity_1_ha"] == areas[i]

    # Finally, with a baseline raster
    fill_raster_lookup = {
        "A": os.path.join(src_dir, "baseline.tif")
    }
    preprocessing._create_value_tables_for_activity(
        sdu_raster_path,
        value_raster_lookup,
        activity_list,
        "activity_1",
        target_folder,
        sdu_list=sdu_list,
        aoi_raster_path=aoi_raster_path,
        mask_raster_path=mask_raster_path,
        fill_raster_lookup=fill_raster_lookup,
        calc_area_for_activity="activity_1"
    )
    df = pd.read_csv(os.path.join(target_folder, "activity_1.csv"))
    df.set_index("SDU_ID", inplace=True)
    vals_a = [28, 32, 35, 40]
    areas = [108, 72, 135, 90]
    for i in range(len(vals_a)):
        assert df.loc[i+1, "A"] == vals_a[i]
        assert df.loc[i+1, "activity_1_ha"] == areas[i]


def test_serviceshed_coverage():

    src_dir = "tests/test_data/dummy_data"

    target_folder = "tests/test_workspace/servicesheds/"
    if not os.path.isdir(target_folder):
        os.makedirs(target_folder)
    
    sdu_raster_path = os.path.join(src_dir, "sdu.tif")
    sdu_list = [1,2,3,4]
    value_raster_lookup = {
        "A": os.path.join(src_dir, "ones.tif")
    }
    activity_list = ["activity_1", "activity_2"]
    preprocessing._create_value_tables_for_activity(
        sdu_raster_path,
        value_raster_lookup,
        activity_list,
        "activity_1",
        target_folder,
        sdu_list=sdu_list,
        calc_area_for_activity="activity_1"
    )

    sdu_grid_path = "tests/test_data/dummy_data/sdu/sdu.shp"
    sdu_id_fieldname = "SDU_ID"
    serviceshed_path_list = ["tests/test_data/dummy_data/servicesheds/ssid.shp"]
    serviceshed_id_list = ["ssid"]
    serviceshed_values = {"ssid": ["w1"]}

    sscov = preprocessing._serviceshed_coverage(
        sdu_grid_path, sdu_id_fieldname, serviceshed_path_list,
        serviceshed_id_list, serviceshed_values
    )

    target_file = "tests/test_workspace/test_create_value_tables_tiny_data/sscov.json"
    with open(target_file, "w") as f:
        json.dump(sscov, f, indent=4)
    
    preprocessing._add_servicesheds(sscov, ["activity_1"], target_folder)

    preprocessing._create_baseline_table(
        os.path.join(target_folder, "activity_1.csv"),
        ["activity_1", "activity_2"],
        ["A"],
        os.path.join(target_folder, "baseline.csv"),
    )    