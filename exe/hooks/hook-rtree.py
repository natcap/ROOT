import os
import glob
from PyInstaller.utils.hooks import \
    (collect_submodules, collect_data_files, get_package_paths)

hiddenimports = collect_submodules('rtree')
datas = collect_data_files('rtree')
