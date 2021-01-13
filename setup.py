from setuptools import setup

setup(
    name = 'rootcode',
    use_scm_version={'version_scheme': 'post-release',
                     'local_scheme': 'node-and-date'},
    packages=[
        'rootcode',
    ],
    package_dir={
        'rootcode': 'rootcode',
    },
    setup_requires=['setuptools_scm'],
    zip_safe=False,
)
