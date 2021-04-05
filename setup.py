from setuptools import setup

setup(
    name = 'natcap.root',
    use_scm_version={'version_scheme': 'post-release',
                     'local_scheme': 'node-and-date'},
    packages=[
        'natcap',
        'natcap.root',
    ],
    package_dir={
        'natcap': 'natcap',
    },
    namespace_packages=['natcap'],
    setup_requires=['setuptools_scm'],
    zip_safe=False,
)
