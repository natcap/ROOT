Documentation for ROOT interface inputs
=======================================

This documentation walks through each of the fields in ROOT's interface, explaining the expected input and choices. Most of the work of setting up a ROOT analysis comes in preparing the input data and configuring a set of tables that describe the analysis to perform. This page describes how to do that.

    .. important::

        * **Rasters**: All rasters must be prepared to have identical extent (bounding boxes) and cell sizes. Please complete this resampling operation using a GIS program or code before running ROOT.
        * **Tables**: All tables must be saved in comma separated variable (csv) format, not xlsx or other spreadsheet formats.

    .. tip::

        Many of the input tables require the user to provide filepaths. Here are easy ways to get these paths:

        * **Windows**: Hold shift and right click the file. Select “Copy as path” in the menu that appears. You can then paste this directly into the csv file.
        * **Mac**: Right click the file, then press and hold the option key. Select “Copy _ as Pathname”, and the text is available to paste into the csv file.


General
-------

* **Workspace**: Folder where results from ROOT will be saved

.. _ig-preprocessing:

Preprocessing
-------------

The preprocessing step prepares the spatial inputs (spatial decision unit map, rasters, and shapefiles) for optimization.

* **Do preprocessing**: If checked, ROOT performs the preprocessing step. If not, ROOT will skip the preprocessing step, which saves time in experimenting with different optimization analyses if preprocessing is already complete.

.. _ig-amt:

* **Activity mask table**: This table points to rasters that indicate valid locations for each activity.

    - *Activity masks*: Rasters with a value of 1 where an activity could take place, and NODATA elsewhere.
    - *Table format*: This table should have headers :attr:`activity` and :attr:`mask_path`. The entries in the first column should be names of each potential activity and the entries in the second column should be complete paths to the corresponding rasters.

        .. csv-table::
            :header: activity, mask_path

            activity1, filepath
            activity2, filepath

.. _ig-iprt:

* **Impact potential raster table**: This table points to the rasters that give the potential impact of each activity on each of the metrics with raster data.

    - *Impact potential raster*: Raster for a particular activity and metric that specifies the value of each pixel being selected for the activity.
    - *Table format*: The table should be a csv file with the following structure:

        .. csv-table::
            :header: activity, factor1, factor2, "..."

            activity1, filepath, filepath, "..."
            activity2, filepath, filepath, "..."

        The upper-left value should be :attr:`activity`, but the other entries should be replaced with corresponding activity names, factor names, or the correct filepath.
    - *Requirements*: Note that if your analysis is using the absolute value approach, you should include an activity called "baseline" (no capitalization), which provides the background value for each factor if no other activity is implemented. If you are using the marginal value approach, do *not* include a "baseline" activity.


* **Spatial weighting maps table**: This table points to shapefiles that specify weighting factors.

    - *Spatial weighting shapefile*: For each named column in a shapefile, ROOT will calculate the average value for each SDU. These values can be used as weighting factors.
    - *Table format*:

        .. csv-table::
            :header: name, file_path, weight_col

            mapname1, filepath, col1 col2 ...
            mapname2, filepath, col1 col2 ...

        In this table, entries in the first column, :attr:`name`, are a user-specified name for the map file. Names for values from each column will be created by combining the map name and column name (e.g. :attr:`mapname1_col1`). Entries in the second column are the path to the shape files. The third column consists of the names of columns of interest from the given shapefile, with the names separated by spaces, *not commas*.

* **Composite factor table**: Allows the user to combine multiple factors using spreadsheet-like expressions to create new ones. These new factors are available to use as constraints or objectives in the optimization step.

    - *Table format*: The table must have columns :attr:`name` and :attr:`formula`:

        .. csv-table::
            :header: name, formula

            new_factor1, f1 * f2
            new_factor2, sqrt(10 \* f3 + 5 \* f4)
    - *Formulas*: The formulas tell ROOT how to combine factors from the raster or shapefile inputs to generate new factors. The new factor is calculated for each SDU and each activity. Any of the basic mathematical operations can be used (+, -, \*, /, ^), as well as numbers, parentheses for grouping, and the functions log, sqrt, and abs. Additionally, sum, min, and max can be used to refer to the corresponding values for a particular factor (*Note*: these are applied separately for each activity - if this is not what you want, you must calculate the overall max yourself).
    - *Activity area* note that preprocessing will create a factor for each activity called :attr:`*activity*_ha` (using the activity names assigned in the activity mask table). These columns can be used in the composite factor table, e.g. to create a cost variable by multiplying by a cost per hectare for the activity.

.. _ig-sdu:

* **Spatial decision unit shape**: Select either a custom SDU shapefile or a regular grid.

    - *Custom shapefile*: in order to use a specific shapefile for the SDUs, enter the path to the file in the textbox. The shapefile must contain a field :attr:`SDU_ID` with unique ID numbers for each SDU polygon.
    - *Regular grid*: in order to have ROOT automatically create an SDU grid, enter either :attr:`square` or :attr:`hexagon` in the text field.

    The SDU shapefile will either be copied or created as sdu_grid.shp in the workspace.

* **Spatial decision unit area**: Specify the area of each SDU polygon for regular grids. Ignored for custom shapefile.

.. _ig-abs-vs-marg:

Absolute vs marginal values
~~~~~~~~~~~~~~~~~~~~~~~~~~~

ROOT offers two modes for evaluation, the first assuming that the impact potential rasters represent "marginal values", meaning the change from the baseline state. The second assuming that they represent "absolute values", meaning they represent the state after the change. In the latter case, ROOT also requires information about the baseline in order to account for the relative changes. In order to do this, there are several specific changes required:

* Provide an activity called "baseline". This activity does not need an activity mask. If you wish to constrain which pixels are aggregated, either provide an area-of-interest shapefile or pre-mask the rasters with other GIS software.
* Provide all impact potential rasters as absolute values.
* ROOT will assess the total values in a given SDU under a certain activity choice by combining the values from the corresponding baseline and activity impact potential rasters - it will assign the activity-specific values to pixels identifed as valid by the corresponding activity mask, and will assign the baseline values to all other pixels. In this way, it captures the change on the relevant pixels and the remaining baseline value on other pixels.



Optimization
------------

* **Do optimization**: If checked, ROOT performs the optimization step.

* **Optimization results suffix**: By default, the results of an optimization run are stored in :attr:`workspace/optimizations`. This field can be used to distinguish results from different runs. If a sufix is provided, the results will be saved to :attr:`workspace/optimizations_suffix`.

.. _ig-optimization-analysis-type:

* **Analysis type**: Tells ROOT which of several optimization analyses to perform. Options are:

    - *weight_table*: Solves one or more optimization runs with user-specified weights assigned to each objective.
    - *n_dim_frontier*: similar to weight_table, except ROOT will randomly generate weights for each objective for each run.

* **Number of frontier points**: Number of optimizations to run (only required for n_dim_frontier analyses)

.. _ig-objectives-table:

* **Objectives table**: This table identifies the factors to optimize for, and additional information depending on the analysis type. For both options, the column headers should be the names of the factors to treat as objectives. Any numeric column from the csv files in :attr:`workspace/sdu_value_tables` can be used. In most cases, these will be the fields named in the tables from the preprocessing steps, although users are free to add additional columns to the SDU value tables containing data from other sources. Note that the columns must be added to the tables for all activities.

    The expected format for each analysis type is:

    - *weight_table*: Each row represents an optimization analysis with particular weights assigned to each factor. Use positive weights to maximize an objective, negative weights to minimize it.

        .. csv-table::
            :header: factor1, factor2, factor3

            w :sub:`11`, w :sub:`12`, w :sub:`13`
            w :sub:`21`, w :sub:`22`, w :sub:`23`

    - *n_dim_frontier*: The table just specifies whether to maximize or minimize each factor:

        .. csv-table::
            :header: factor1, factor2, factor3

            min, min, max

.. _ig-targets_table:

* **Targets table**: Allows the user to set targets (constraints) for the optimizations. The table should have columns :attr:`formula`, :attr:`cons_type`, and :attr:`value`.

        .. csv-table::
            :header: formula, cons_type, value

            f1 + f2 + f3, <=, *budget*
            f4 + f5, >=, *target*
            f6, >= *target*

    - *formula*: An expression following the same rules as the expressions for the Composite Factor Table.
    - *cons_type*: one of =, <=, or >=.
    - *value*: the numerical value for the target (constraint).