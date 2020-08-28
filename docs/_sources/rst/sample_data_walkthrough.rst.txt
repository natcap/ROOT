Sample data walkthrough
=======================

Download sample data here:


Context
-------
The sample data comes from an analysis of different restoration or agricultural management changes in a watershed in Colombia. The goal is to determine which areas to target with each activity in order to achieve desired benefits to carbon sequestration, nitrogen and sediment load reduction, and improvements to water supply. Note that the models were not calibrated, so this example should be considered illustrative, not prescriptive.

The activities in the example are:

* **Ag BMPS**: This activity represents a shift from a conventional-style agriculture to one employing improved nutrient management techniques.
* **Forest restoration**: This activity captures a significant change in land use from crop or pasture to a restored forest.
* **Riparian restoration**: This activity is similar to forest restoration, but is specific to a buffer around waterways to target critical areas for water quality.

And the biophysical factors, measured as change from baseline to alternative activity, are:

* **Carbon**: change in carbon storage (soil and biomass)
* **NDR**: change in nitrogen loading
* **SDR**: change in sediment loading
* **SWY**: change in seasonal water yield


File overview
-------------
The input data comes in four folders:

* **activity_mask_rasters**: These rasters identify which pixels are valid locations for each of the activities.
* **configuration_tables**: This folder contains the various tables that will go into the fields in the ROOT interface. They are numbered in the order that they appear in the interface. As you work through the example, you will need to edit these tables to adjust file paths and set various options. Feel free to make copies of the files before editing them - the names of the files are not important to ROOT, so you can name them whatever is helpful to you.
* **impact_potential_rasters**: These rasters contain the pixel-level values of each activity for each of the biophysical factors. They are named mv_*factor*_*activity*.tif, where mv is short for "marginal value", and factor and activity are replaced with the corresponding names. So, for example, "mv_ndr_forest_restoration.tif" gives the effect of forest restoration on nitrogen loading.
* **spatial_weighting_maps**: This map indicates some population characteristics that could be used for targeting different activities or objectives.

Configuring Preprocessing tables
--------------------------------
Since the data is provided ready to go, the first step will be to set up the four tables that configure the preprocessing step of the analysis. These are:

* Activity mask table
* Impact potential raster table
* Spatial weighting map table
* Composite factor table

Each of them will need to be configured appropriately to run the analysis. We will go through each one in order. Templates are provided in the `configuration_tables` folder, with numeric prefixes to line them up in the order we will handle them (the same as their order in the UI).

Activity mask table
~~~~~~~~~~~~~~~~~~~
This table tells ROOT which rasters to use to identify pixels that are potential sites for each activity. To get it ready to use:

1) Open 1_am_table.csv. Note it has two columns, 'activity', and 'mask_path'. The 'activity' column is where you specify a name to use for each activity in the analysis. In the example data, it has been filled in already.
2) For each activity, find the appropriate raster in `activity_mask_rasters`. Copy the complete file path (see the tips in the interface guide for how to do this easily) and paste it in the corresponding space in the 'mask_path' column.
3) Save and close the table. Say 'yes' to keeping the .csv format.

Impact potential raster table
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This table tells ROOT which rasters provide spatial data on the impact of each practice for a given set of measures.

1) Open 2_ipm_table.csv. Note it has a column called 'name', and then one column for each of the activities named in the activity mask table. In the example data, these activity names have been filled in already, but in other cases, you will have to make sure to transfer the names correctly between tables (paste special -> transpose can help avoid typos). In the 'name' column, the sample table includes names of the different metrics we will provide data for.
2) For each activity and metric, find the raster in the `impact_potential_rasters` folder. The files are named for both the factor and the activity (see above). Copy the file paths into the correct cells in the table.
3) Save and close the table.

Spatial weighting maps table
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This table tells ROOT about shapefiles that provide additional weighting information. It lists the files and which fields from the attribute tables to pull out.

1) Open 3_swm_table.csv. It has 3 columns, one to assign a name to each shapefile, one for the filepath, and one to identify the fields to keep.
2) For this example, we only have one shapefile, so copy the filepath of the `pop_chars.shp` file (found in `spatial_weighting_maps`) into the file_path field in the csv.
3) Type the following into the weight_col field: ``prop_poor prop_dep_o``. Note it is important to separate the names of the fields using a space, not a comma.
4) Save and close the table.

Composite factors table
~~~~~~~~~~~~~~~~~~~~~~~
This table tells ROOT how to combine the different factors from the raster and shapefile data to create composite values. This example will use simple combinations, but more complicated mathematical formulas can be used.

1) Open 4_cf_table.csv. The first column, 'name', allows you to set the name for the composite factor, and the second, 'factors', lets you write the formula ROOT will use to calculate it.
2) In this example, we have provided an example for a poverty-weighted nitrogen metric. Fill in the empty box replicating the ndr equation, but replacing ndr with sdr.
3) Save and close the table.


Setting up the preprocessing UI inputs
--------------------------------------
This section will go through running ROOT's preprocessing step.

1) Open ROOT. To do this, double-click on the 'ROOT.exe' file.
2) Pick a workspace. This is where the results of the analysis will be saved. Click on the folder icon to navigate to the desired folder.
3) Check the "Do Preprocessing" box, and make sure the "Do Optimization" box is unchecked.
4) For each of the first four fields in the "Preprocessing Arguments" section, click on the folder icon and navigate to the corresponding table in the `configuration_tables` folder.
5) Enter ``hexagon`` for the spatial decision unit shape and ``100.0`` for the area in the next two fields.
6) Click Run to perform the preprocessing

    .. error::
        If there are errors, it is likely that a filepath has been entered incorrectly or activity/factor names don't match between files. Read the error messages and check the configuration tables to troubleshoot until the preprocessing runs.

Preprocessing outputs
---------------------
This preprocessing will create two types of files in the workspace folder:

* **sdu_grid.shp**: This is an automatically generated shapefile that outlines the spatial decision units (SDUs) for the problem. It is based on the SDU shape and area arguments to the UI, so in this case consists of 100 hectare hexagons.
* **activity tables in sdu_value_tables**: These tables provide summaries of the spatial input data aggregated to the SDUs. The columns include information on the available area for each activity by SDU, summed values for each of the raster-based factors, average values for the shapefile-based weights, and results of applying the composite factor expressions. These tables are the data that will be used by the optimization step to assign activities to SDUs.


Optimization examples
---------------------
