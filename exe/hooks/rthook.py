import multiprocessing
import os
import platform
import sys

multiprocessing.freeze_support()

os.environ['MATPLOTLIBDATA'] = os.path.join(sys._MEIPASS, 'mpl-data')

os.environ['PROJ_LIB'] = os.path.join(sys._MEIPASS, 'proj')

# libspatialindex is put into the binary's root directory
os.environ['SPATIALINDEX_C_LIBRARY'] = sys._MEIPASS

if platform.system() == 'Darwin':
    os.environ['GDAL_DATA'] = os.path.join(sys._MEIPASS, 'gdal-data', 'gdal')

    # This allows Qt 5.13+ to start on Big Sur.
    # See https://bugreports.qt.io/browse/QTBUG-87014
    # and https://github.com/natcap/invest/issues/384
    os.environ['QT_MAC_WANTS_LAYER'] = '1'
