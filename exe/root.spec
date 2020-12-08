# -*- mode: python -*-
import os
import sys
import shutil
from PyInstaller.compat import is_win

block_cipher = None

path_extension = []
conda_env = os.environ['CONDA_PREFIX']

if is_win:
    proj_datas = ((os.path.join(conda_env, 'Library/share/proj'), 'proj'))
else:
    proj_datas = ((os.path.join(conda_env, 'share/proj'), 'proj'))

# Add the root_ui directory to the extended path.
path_extension.insert(0, os.path.abspath('..'))

a = Analysis([os.path.join('..', 'rootcode', 'root.py')],
             pathex=path_extension,
             binaries=None,
             datas=[('qt.conf', '.'), proj_datas],
             hiddenimports=[
                'pygeoprocessing',
                'root',
                'distutils',
                'distutils.dist',
                'distutils.version',
                'natcap.invest',
                'natcap.invest.ui',
                'natcap.invest.ui.launcher',
                'osgeo.gdal',
                'osgeo._gdal',
                'shapely',
                'rtree',
                'pandas._libs.skiplist',
                'scipy._lib.messagestream',
             ],
             hookspath=[os.path.join(os.getcwd(), 'exe', 'hooks')],
             runtime_hooks=None,
             excludes=None,
             win_no_prefer_redirects=None,
             win_private_assemblies=None,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='root',
          debug=False,
          strip=None,
          upx=True,
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=None,
               upx=True,
               name='root')

#if is_win:
#    # For some reason, the _gdal dylib isn't copied to the correct name.
#    # Copy-pasting like this feels like a hack, but it should work ok.
#    bindir = os.path.join('dist', 'root-x64', 'root')
#    shutil.copyfile(os.path.join(bindir, 'osgeo._gdal.pyd'),
#                    os.path.join(bindir, '_gdal.pyd'))
