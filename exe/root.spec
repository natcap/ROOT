# -*- mode: python -*-
import os
import sys
import shutil
import glob
import itertools
from PyInstaller.compat import is_win, is_darwin

block_cipher = None

path_extension = ['rootcode']
conda_env = os.environ['CONDA_PREFIX']

if is_win:
    proj_datas = ((os.path.join(conda_env, 'Library/share/proj'), 'proj'))
else:
    proj_datas = ((os.path.join(conda_env, 'share/proj'), 'proj'))

# Add the root_ui directory to the extended path.
path_extension.insert(0, os.path.abspath('..'))

a = Analysis([os.path.join(os.getcwd(), 'natcap', 'root', 'root.py')],  # Assume we're building from the project root
             pathex=path_extension,
             binaries=None,
             datas=[('qt.conf', '.'), proj_datas],
             hiddenimports=[
                'pygeoprocessing',
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
                'scipy.special.cython_special',
                'scipy.spatial.transform._rotation_groups',
                'cmath',
             ],
             hookspath=[os.path.join(os.getcwd(), 'exe', 'hooks')],
             runtime_hooks=[os.path.join(
                os.getcwd(), 'exe', 'hooks', 'rthook.py')],
             excludes=None,
             win_no_prefer_redirects=None,
             win_private_assemblies=None,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

# Create the executable file.
if is_darwin:
    # add rtree, shapely, proj dependency dynamic libraries from conda
    # environment.
    # These libraries are specifically included here because they don't seem to
    # be picked up by the built-in hooks and have been known to interfere with
    # the pyinstaller installation when running on a homebrew-enabled system.
    # See https://github.com/natcap/invest/issues/10.
    a.binaries += [
        (os.path.basename(name), name, 'BINARY') for name in
        itertools.chain(
            glob.glob(os.path.join(conda_env, 'lib', 'libspatialindex*.dylib')),
            glob.glob(os.path.join(conda_env, 'lib', 'libgeos*.dylib')),
            glob.glob(os.path.join(conda_env, 'lib', 'libproj*.dylib')),
        )
    ]
elif is_win:
    # Adapted from
    # https://shanetully.com/2013/08/cross-platform-deployment-of-python-applications-with-pyinstaller/
    # Supposed to gather the mscvr/p DLLs from the local system before
    # packaging.  Skirts the issue of us needing to keep them under version
    # control.
    a.binaries += [
        ('msvcp90.dll', 'C:\\Windows\\System32\\msvcp90.dll', 'BINARY'),
        ('msvcr90.dll', 'C:\\Windows\\System32\\msvcr90.dll', 'BINARY')
    ]

    # add rtree dependency dynamic libraries from conda environment
    a.binaries += [
        (os.path.basename(name), name, 'BINARY') for name in
        glob.glob(os.path.join(conda_env, 'Library/bin/spatialindex*.dll'))]

exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='root',
          debug=False,
          strip=None,
          upx=True,
          console=True)
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=None,
               upx=True,
               name='root')
