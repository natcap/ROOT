import os
import sys
import shutil
import math
import tempfile
import collections
import glob
import shapely.wkb
import shapely.prepared

import numpy
from osgeo import ogr
from osgeo import gdal
from osgeo import osr
import numpy as np
import pandas as pd

import pygeoprocessing
from natcap.root import arith_parser as ap


class RootPreprocessingError(Exception):
    pass


def _clean_negative_nodata_values(
        base_raster_path, target_clean_raster_path):
    """Reset large negative corrupt nodata values to valid ones.

    Parameters:
        base_raster_path (string): path to a single band floating point
            raster with a large negative nodata value and pixel values that
            might also  be nodata but are corrupt from roundoff error.
        target_clean_raster_path (string): path to desired target raster that
            will ensure the nodata value is a ffinto(float32).min and any
            values in the source raster that are close to that value are set
            to this.

    Returns:
        None.
    """
    target_nodata = np.finfo(np.float32).min
    raster_info = pygeoprocessing.get_raster_info(base_raster_path)
    base_nodata = raster_info['nodata'][0]
    if base_nodata > target_nodata / 10:
        print(
            "Base raster doesn't have a large nodata value; it's likely"
            " not one of the corrupt float32.min rasters we were dealing"
            " with.  Copying %s to %s without modification." % (
                base_raster_path, target_clean_raster_path))
        shutil.copy(base_raster_path, target_clean_raster_path)

    pygeoprocessing.new_raster_from_base(
        base_raster_path, target_clean_raster_path, gdal.GDT_Float32,
        [target_nodata])
    target_raster = gdal.Open(target_clean_raster_path, gdal.GA_Update)
    target_band = target_raster.GetRasterBand(1)

    for block_offset, data_block in pygeoprocessing.iterblocks(
            (base_raster_path, 1)):
        possible_nodata_mask = data_block < (target_nodata / 10)
        data_block[possible_nodata_mask] = target_nodata
        target_band.WriteArray(
            data_block, xoff=block_offset['xoff'], yoff=block_offset['yoff'])


def _aggregate_raster_values(
        sdu_raster_path, sdu_grid_path, sdu_key_id, mask_raster_path,
        value_raster_lookup, baseline_raster_lookup=None):
    """Build table that indexes SDU ids with aggregated marginal values.

    Parameters:
        sdu_grid_path (string): path to single layer polygon vector with
            integer field id that uniquely identifies each polygon.
        sdu_key_id (string): field in `sdu_grid_path` that uniquely identifies
            each feature.
        mask_raster_path (string): path to a mask raster whose pixels are
            considered "valid" if they are not nodata.
        value_raster_lookup (dict): keys are marginal value IDs that
            will be used in the optimization table; values are paths to
            single band rasters.
        baseline_raster_lookup (dict): same as `value_raster_lookup` but
            for the baseline rasters. Need this if calculating "merged" values.

    Returns:
        A dictionary that encapsulates stats about each polygon, mask coverage
        and marginal value aggregation and coverage. Each key in the dict is
        the SDU_ID for a polygon, while the value is a tuple that contains
        first polygon/mask stats, then another dict for marginal value stats.
        In pseudocode:
            { sdu_id0:
                (sdu area, sdu pixel coverage, mask pixel count,
                 mask pixel coverage in Ha),
                {marginal value id a: (
                    aggregated values, n pixels of coverage,
                    aggregated value per Ha of coverage),
                 marginal valud id b: ...},
              sdu_id1: ...
            }
    """
    # TODO: drop activity mask, get activity from the rasters - require nodata for non-transition pixels

    print('marginal_value_lookup: {}'.format(value_raster_lookup))
    value_ids = value_raster_lookup.keys()

    value_rasters = [
        gdal.Open(value_raster_lookup[value_id])
        for value_id in value_ids]
    value_bands = [
        raster.GetRasterBand(1) for raster in value_rasters]
    value_nodata_list = [
        band.GetNoDataValue() for band in value_bands]
    
    if baseline_raster_lookup is not None:
        baseline_value_rasters = [
            gdal.Open(baseline_raster_lookup[value_id])
            for value_id in value_ids]
        baseline_value_bands = [
            raster.GetRasterBand(1) for raster in baseline_value_rasters]
        baseline_value_nodata_list = [
            band.GetNoDataValue() for band in baseline_value_bands]

    mask_raster = gdal.Open(mask_raster_path)
    mask_band = mask_raster.GetRasterBand(1)
    mask_nodata = mask_band.GetNoDataValue()
    geotransform = mask_raster.GetGeoTransform()
    # note: i'm assuming square pixels that are aligned NS and EW and
    # projected in meters as linear units
    pixel_area_m2 = float((geotransform[1]) ** 2)

    id_raster = gdal.Open(sdu_raster_path)
    id_band = id_raster.GetRasterBand(1)
    id_nodata = id_band.GetNoDataValue()
    id_band = None
    id_raster = None

    # first element in tuple is the coverage stats:
    # (sdu area, sdu pixel count, mask pixel count, mask pixel coverage in Ha)
    # second element 3 element list (aggregate sum, pixel count, sum/Ha)
    value_sums = collections.defaultdict(
        lambda: (
            [0.0, 0, 0, 0.0],
            dict((mv_id, [0.0, 0, None]) for mv_id in value_ids)))

    # format of sdu_coverage is:
    # (sdu area, sdu pixel count, mask pixel count, mask pixel coverage in Ha)
    for block_offset, id_block in pygeoprocessing.iterblocks(
            (sdu_raster_path, 1)):
        value_blocks = [
            band.ReadAsArray(**block_offset) for band in value_bands]
        if baseline_raster_lookup is not None:
            baseline_value_blocks = [
                band.ReadAsArray(**block_offset) for band in baseline_value_bands]
        else:
            baseline_value_nodata_list = [None for _ in value_bands]
            baseline_value_blocks = [None for _ in value_bands]
        
        mask_block = mask_band.ReadAsArray(**block_offset)
        
        for aggregate_id in np.unique(id_block):
            if aggregate_id == id_nodata:
                continue
            aggregate_mask = id_block == aggregate_id
            # update sdu pixel coverage
            # value_sums[aggregate_id][0] =
            #    (sdu area, sdu pixel count, mask pixel count, mask pixel Ha)
            value_sums[aggregate_id][0][1] += np.count_nonzero(
                aggregate_mask)
            valid_mask_block = mask_block[aggregate_mask]
            value_sums[aggregate_id][0][2] += np.count_nonzero(
                valid_mask_block != mask_nodata)
            for mv_id, mv_nodata, mv_block, base_nodata, base_block in zip(
                    value_ids, value_nodata_list,
                    value_blocks, baseline_value_nodata_list, baseline_value_blocks):
                valid_mv_block = mv_block[aggregate_mask]
                # raw aggregation of marginal value
                # value_sums[aggregate_id][1][mv_id] =
                # (sum, pixel count, pixel Ha)
                value_sums[aggregate_id][1][mv_id][0] += np.nansum(
                    valid_mv_block[np.logical_and(
                        valid_mv_block != mv_nodata,
                        valid_mask_block != mask_nodata)])
                if base_block is not None:
                    value_sums[aggregate_id][1][mv_id][0] += np.nansum(
                        base_block[np.logical_and(
                            base_block != base_nodata,
                            valid_mask_block == mask_nodata)])

                # pixel count coverage of marginal value
                value_sums[aggregate_id][1][mv_id][1] += (
                    np.count_nonzero(np.logical_and(
                        valid_mv_block != mv_nodata,
                        valid_mask_block != mask_nodata)))
    
    # calculate SDU, mask coverage in Ha, and marginal value Ha coverage
    for sdu_id in value_sums:
        value_sums[sdu_id][0][0] = (
            value_sums[sdu_id][0][1] * pixel_area_m2 / 10000.0)
        value_sums[sdu_id][0][3] = (
            value_sums[sdu_id][0][2] * pixel_area_m2 / 10000.0)
        # calculate the 3rd tuple of marginal value per Ha
        for mv_id in value_sums[sdu_id][1]:
            if value_sums[sdu_id][1][mv_id][1] != 0:
                value_sums[sdu_id][1][mv_id][2] = (
                    value_sums[sdu_id][1][mv_id][0] / (
                        value_sums[sdu_id][1][mv_id][1] *
                        pixel_area_m2 / 10000.0))
            else:
                value_sums[sdu_id][1][mv_id][2] = 0.0
    
    del value_bands[:]
    del value_rasters[:]
    mask_band = None
    mask_raster = None
    return value_sums


def _build_ip_table(
        sdu_col_name, activity_list, activity_name, marginal_value_lookup,
        sdu_serviceshed_coverage, target_ip_table_path, baseline_table=False):
    """Build a table for Integer Programmer.

    Output is a CSV table with columns identifying the aggregating SDU_ID,
    stats about SDU and mask coverage, as well as aggregate values for
    marginal values.

    Parameters:
        sdu_col_name (string): desired name of the SDU id column in the
            target IP table.
        marginal_value_lookup (dict): in pseudocode:
         { sdu_id0:
                (sdu area, sdu pixel coverage, mask pixel count,
                 mask pixel coverage in Ha),
                {marginal value id a: (
                    aggreated values, n pixels of coverage,
                    aggregated value per Ha of covrage),
                 marginal value id b: ...},
              sdu_id1: ...
            }
        sdu_serviceshed_coverage (dict): in pseudocode:
            {
                sdu_id_0: {
                    "serviceshed_id_a":
                        [serviceshed coverage proportion for a on id_0,
                         {service_shed_a_value_i: sum of value_i multiplied
                          by proportion of coverage of sdu_id_0 with
                          serviceshed_id_a.}]
                    "serviceshed_id_b": ....
                },
                sdu_id_1: {....
            }
        target_ip_table_path (string): path to target IP table that will
            have the columns:
                SDU_ID,pixel_count,area_ha,maskpixct,maskpixha,mv_ida,mv_ida_perHA
    """
    if activity_name is not None:
        try:
            activity_index = activity_list.index(activity_name)
        except ValueError:
            msg = 'activity_name not found in activity_list in _build_ip_table'
            raise RootPreprocessingError(msg)
    else:
        activity_index = None

    with open(target_ip_table_path, 'w') as target_ip_file:
        # write header
        target_ip_file.write(
            "{},pixel_count,area_ha".format(sdu_col_name))
        target_ip_file.write(",%s_ha" * len(activity_list) % tuple(activity_list))
        target_ip_file.write(',exclude')
        # target_ip_file.write(
        #     "{},pixel_count,area_ha,{}_px,{}_ha".format(
        #         sdu_col_name, activity_name, activity_name))
        # This gets the "first" value in the dict, then the keys of that dict
        # also makes sense to sort them so it's easy to navigate the CSV.
        marginal_value_ids = sorted(
            list(marginal_value_lookup.values())[0][1].keys())
        n_mv_ids = len(marginal_value_ids)
        target_ip_file.write((",%s" * n_mv_ids) % tuple(marginal_value_ids))
        # target_ip_file.write(
        #     (",%s_perHA" * n_mv_ids) % tuple(marginal_value_ids))
        if sdu_serviceshed_coverage is not None:
            first_serviceshed_lookup = list(sdu_serviceshed_coverage.values())[0]
        else:
            first_serviceshed_lookup = {}
        serviceshed_ids = sorted(first_serviceshed_lookup.keys())
        target_ip_file.write(
            (",%s" * len(serviceshed_ids)) % tuple(serviceshed_ids))
        value_ids = {
            sid: sorted(first_serviceshed_lookup[sid][1].keys()) for
            sid in serviceshed_ids
            }
        for serviceshed_id in serviceshed_ids:
            for value_id in value_ids[serviceshed_id]:
                target_ip_file.write(",%s_%s" % (serviceshed_id, value_id))
        target_ip_file.write('\n')

        # write each row
        for sdu_id in sorted(marginal_value_lookup):
            # id, pixel count, total pixel area,
            target_ip_file.write(
                "%d,%d,%f" % (
                    sdu_id, marginal_value_lookup[sdu_id][0][1],
                    marginal_value_lookup[sdu_id][0][0]))

            # areas by activity
            areas = [0 for _ in range(len(activity_list))]
            if baseline_table is False and activity_index is not None:
                areas[activity_index] = marginal_value_lookup[sdu_id][0][3]
            target_ip_file.write(",%f" * len(areas) % tuple(areas))
            # if all areas are 0, that means in particular the current activity has 0 available area
            # and we want to exclude this SDU as an option
            if baseline_table is False and max(areas) == 0:
                target_ip_file.write(',1')
            else:
                target_ip_file.write(',0')

            # write out all the marginal value aggregate values
            for mv_id in marginal_value_ids:
                target_ip_file.write(
                    ",%f" % marginal_value_lookup[sdu_id][1][mv_id][0])
            # write out all marginal value aggregate values per Ha
            # for mv_id in marginal_value_ids:
            #     target_ip_file.write(
            #         ",%f" % marginal_value_lookup[sdu_id][1][mv_id][2])
            # serviceshed values
            for serviceshed_id in serviceshed_ids:
                target_ip_file.write(
                    (",%f" % sdu_serviceshed_coverage[sdu_id][serviceshed_id][0]))
            for serviceshed_id in serviceshed_ids:
                for value_id in value_ids[serviceshed_id]:
                    target_ip_file.write(
                        (",%f" % sdu_serviceshed_coverage[sdu_id][serviceshed_id][1][value_id]))
            target_ip_file.write('\n')
