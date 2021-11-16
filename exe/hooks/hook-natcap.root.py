from PyInstaller.utils.hooks import (
    collect_submodules, collect_data_files, copy_metadata)

datas = collect_data_files('natcap.root') + copy_metadata('natcap.root')
hiddenimports = collect_submodules('natcap.root')
