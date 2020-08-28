from PyInstaller.utils.hooks import (
    collect_submodules, collect_data_files, copy_metadata)

datas = collect_data_files('natcap.invest') + copy_metadata('natcap.invest')
hiddenimports = collect_submodules('natcap.invest')
