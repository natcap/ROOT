ROOT Workflow
=============

This section describes the steps in the ROOT analysis. Running ROOT requires a fair amount of up-front work assembling spatial data to use as inputs to the optimization. Once that data is in place, the pre-processing step in ROOT translates the input data from raster and shapefile formats to a table-based format for the optimization. An analysis will then typically involve a number of different optimization steps based on the same pre-processed data to compare use of different objectives, weights, and constraint values.


Key terms
----------
.. glossary::

    Activity
        ROOT is used to inform spatial decisions about where to undertake different management actions or changes. We use the term "activity" to refer generally to any action, investment, or decision that needs to be spatially allocated. The result of a single optimization from ROOT will be a table indicating which activities, if any, to assign to each spatial decision unit. Some of these activities may not require any physical action (e.g. allocation of protection to natural areas).

    Factor
        The analyses that ROOT performs are based on characteristics of each spatial decision unit and activity that determine whether an activity allocation is good for the specified objectives, or obeys rules (constraints) that the decision is subject to. We refer generally to any of these characteristics as a 'factor'. Specifically, factors show up as the columns in the tables output from the preprocessing steps, and are calculated from the raster and shape inputs.

    Spatial decision unit (SDU)
        Rather than making pixel-level decisions, ROOT optimizes activity allocation at a more aggregated spatial scale, called the spatial decision unit (SDU). These can be customized by the user, or ROOT can create a grid of equal-area squares or hexagons.

    Agreement map
        A full run of ROOT generally entails many separate optimizations, each with different importances assigned to the objectives. The agreement map is a summary output that indicates how often the same allocation is made regardless of the particular weights. This allows the user to identify allocations that make sense across a wide range of preferences and those that are much more sensitive.


Data preparation
----------------

* Creating the IPRs
* Creating other maps
* make sure the rasters match

Preprocessing
--------------
The preprocessing step converts the spatial input data to a tabular form that summarizes the input data at the SDU level, as needed by the optimization step. This section describes each of the types of data inputs and how they are used.

SDU map
~~~~~~~~
This is a shapefile that outlines the spatial decision units (SDUs) for the analysis. The preprocessing step involves aggregating the values from the raster and shapefile inputs to provide scores for each activity for each SDU.

The map of spatial decision units can either be provided by the user as a custom map, or generated automatically by ROOT as a regular grid of squares or hexagons. A user provided map makes sense when there are relevant decision boundaries, e.g. subwatershed outlines, a list of proposed projects, or political/ownership parcels. Note that the SDU map must have a field called SDU_ID with unique values for each SDU. For a more generalized analysis, ROOT can generate a map of equal-area squares or hexagons.

Activity boundary rasters
~~~~~~~~~~~~~~~~~~~~~~~~~
These are rasters, one per activity, that indicate potential locations for that activity. When a particular activity is allocated to an SDU, ROOT will treat it as applying only the pixels (grid cells) indicated by the activity mask. That means only the values from the impact potential rasters that overlap with the potential activity pixels will be counted.

The criteria used to construct the activity boundary rasters will depend on the decision context and the activity in question.

Impact potential rasters
~~~~~~~~~~~~~~~~~~~~~~~~

Raster data captures spatially-explicit value of a given activity towards a given factor. For each raster, the preprocessing step will calculate the total score per SDU by summing all the pixels that overlap with allowed pixels in the corresponding mask.

The impact potential rasters can be generated in many different ways, and ROOT places no requirements on the source of the data. One important consideration is whether to use absolute or relative values. For example, for the service of carbon storage, the rasters could either contain values of expected mass of carbon for each land use, or contain the expected change from baseline for each activity. It is probably easier to choose one approach and apply it across all factors for the sake of consistency, but if a problem is more suited to a mixture of approaches, ROOT will still operate correctly.

**Figure: Raster data + Activity Mask + SDU grid --> activity value table**

Shapefiles
~~~~~~~~~~

Shapefiles, called spatial weighting maps (SWMs), are used to provide non-activity related data, such as defining spatially dependent weights for different factors, or defining regions for setting targets or constraints. For each shapefile, the user will indicate which fields to aggregate. For each named field, the preprocessing step will calculate the average value of features in the SWM overlapping with each SDU.

Output
~~~~~~
When the preprocessing step runs, it creates a set of tables that are saved in a folder called "sdu_value_tables" inside the project workspace. There will be one table for each activity plus one called "baseline.csv". Each table contains the per-SDU scores for each for the factors identified in the IPRs, SWMs, and CFT for a single activity. The tables are indexed by the SDU_ID column, so it is possible to join them with the SDU map to visualize and compare the SDU-level values.


Optimization
------------
This phase takes a description of the optimization problem and performs a number of separate optimizations to generate a portfolio of solutions. In the "standard" ROOT analysis, each of these optimizations assigns a different level of importance to each objective, so the full set of analyses generates a production possibility frontier (trade-off curve). Users can then see how different objectives trade off with each other across the range of possible solutions.

Mathematical Framework
~~~~~~~~~~~~~~~~~~~~~~
The ROOT UI uses a linear programming approach, which is a particular way of formulating optimization problems where the variables are continuous, and the objectives and constraints are made up of linear functions (note: if ROOT is used through the python interface, it is also possible to specify the use of integer variables).

In each optimization, ROOT will specify what fraction of each SDU's potential area for each activity should be selected. Unless specified otherwise, these choices are assumed to be exclusive of each other within an SDU. The decision variables for the optimization are :math:`x_{sa}`, the fraction of available area for activity *a* to allocate in sdu *s*.

ROOT uses a weighted-sum formulation to combine different factors into a single objective function. For each optimization, weights :math:`w_i` are assigned to each sub-objective, either randomly or specified by the user. The values for the sub-objectives come from the tables generated by the preprocessing step, and depend on the SDU and activity. Any named factor taken from the input rasters, shapefiles, or constructed through the combined factor table can be used as a sub-objective. The objective function for that optimization is:

.. math:: \max_{x_{sa}} \sum_i w_i V_{isa} x_{sa}

subject to whatever constraints are specified for the analysis, where :math:`V_{isa}` represents the value to sub-objective *i* of selecting 100% of potential area for activity *a* in SDU *s*. By varying the weights, ROOT can find solutions across the production possibility frontier, or with user-provided weights, ROOT allows specific analyses.

Problem types
~~~~~~~~~~~~~
The first step is deciding what problem type to use. ROOT offers two primary modes, an automatic random weight sampling and use of weights from a user-provided table. There are a number of ways these modes can be used to facilitate different kinds of analyses. In a basic case, the random sampling mode may be all that is needed to outline the production frontier and generate an agreement map. In other cases, users may want to generate a table of weights following a particular sampling scheme (e.g. a space-filling routine), or to perform sensitivity analyses around particular weight values.

Objective
~~~~~~~~~
Defining the objective function in ROOT involves identifying which of the factors from the preprocessing step should be maximized or minimized. In the random weight mode, the user simply species the factors and whether each should be maximized or minimized. ROOT then generates a number of solutions using random weights for each sub-objective in each analysis. In the weight table mode, the user provides a table that specifies the particular values of :math:`w_i` to assign to each sub-objective for each iteration.

Constraints
~~~~~~~~~~~
Most planning problems have constraints of one type or another that can be included in the ROOT analysis. Some basic examples are a fixed budget, target areas for a land-based activity, or minimum values of an objective. These constraints have the mathematical form

.. math:: \sum_{sa} V_{sa} x_{sa} \ge, =, \le T

which compares the total value across the activity allocation of a particular factor against a target value. More complicated constraints that combine multiple factors are also possible (see the interface guide for how to use expressions to define multi-factor constraints).


Outputs
--------
The outputs from ROOT consist tables for the solution (an allocation of activities to each SDU, :math:`x_{sa}`) for each iteration, an agreement map that provides a spatial summary of the consistency of the activty allocations, and a summary table that provides total sub-objective values for each solution and can be used to plot the production possibility frontier.

