"""Performs post-optimization analysis.
"""

import os

from osgeo import ogr
from osgeo import osr
import pandas as pd


def execute(args):
    """Constructs agreement map table and shapefile.

    Args:
        args (dict): ROOT problem configuration dict

    Returns:
        None
    """

    solution_folder = args['optimization_folder']
    agreement_map_files = os.path.join(solution_folder, 'summed_choices.csv')
    adf = pd.read_csv(agreement_map_files)
    activity_names = adf.columns[1:]

    def trim_to_10chars(s):
        if len(s) <= 10:
            return s
        else:
            return s[:10]

    field_names = {a: trim_to_10chars(a) for a in activity_names}
    adf.set_index('SDU_ID', inplace=True)

    # duplicate sdu_grid to optimizations/agreement.shp adding agreement column
    driver = ogr.GetDriverByName('ESRI Shapefile')
    sdu_file_path = os.path.join(args['workspace'], 'sdu_grid.shp')
    target_file_path = os.path.join(solution_folder, 'agreement_map.shp')

    # open base grid file
    sdu_file = ogr.Open(sdu_file_path)
    orig_layer = sdu_file.GetLayer()
    spatial_ref = osr.SpatialReference(orig_layer.GetSpatialRef().ExportToWkt())

    # create new grid shapefile and define fields
    if os.path.exists(target_file_path):
        driver.DeleteDataSource(target_file_path)
    target_file = driver.CreateDataSource(target_file_path)
    target_layer = target_file.CreateLayer(
        'choice_ct', spatial_ref, ogr.wkbPolygon
    )
    target_layer.CreateField(
        ogr.FieldDefn('SDU_ID', ogr.OFTInteger)
    )
    for activity in activity_names:
        target_layer.CreateField(
            ogr.FieldDefn(field_names[activity], ogr.OFTReal)
        )
    target_layer_defn = target_layer.GetLayerDefn()

    # loop through features in original grid
    # add agreement score and geoms to target layer
    for feature in orig_layer:
        feature_id = int(feature.GetField('SDU_ID'))
        if feature_id not in adf.index:
            continue

        geom = feature.GetGeometryRef()

        new_feature = ogr.Feature(target_layer_defn)
        new_feature.SetGeometry(geom)
        new_feature.SetField('SDU_ID', feature_id)
        for activity in activity_names:
            score = adf.loc[feature_id][activity]
            new_feature.SetField(field_names[activity], score)
        target_layer.CreateFeature(new_feature)

    target_layer.SyncToDisk()