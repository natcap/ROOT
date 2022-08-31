Release Notes
=============


1.2 (Aug 2022)
--------------
Introduces new features:

* **Absolute value analysis**: Instead of requiring "marginal values" for the optimization, ROOT will now function with "absolute values" with the inclusion of a baseline scenario.
* **Area of interest**: A new option to provide an area of interest map has been added. This allows the user to use a shapefile as an additional boundary to determine the area to analyze.
* **Advanced options**: Some new advanced options are now accessible by providing a json-formatted options file.

Internal improvements:

* **Preprocessing refactoring**: Various changes were made to the ROOT preprocessing steps to  improve the analysis and facilitate future development


1.1 (July 2018)
---------------
Introduces new features:

* **Multiple activities**: ROOT can now optimize locations for more than one class of activity in the same analysis.
* **Combined factor and constraint formulas**: Users can now use mathematical formulas to define new factors (objectives or constraints) from those provided by the raster and shapefile inputs. Previously new factors could only be created by multiplying.
* **Custom SDU shapefiles**: ROOT now provides users the option of providing their own spatial decision unit shapefile to use in place of a regular grid.
* **Specific weight optimizations**: Users can run a set of optimization analyses with specific weights on sub-objectives by providing a table listing the desired weights.
* **Input validation**: Inspects input tables and provides specific error messages.
* **Other**: Various naming changes to make terminology and field names more consistent.


1.0 - Initial release
---------------------
Initial release of ROOT.