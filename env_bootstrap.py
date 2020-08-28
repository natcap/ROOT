import virtualenv, textwrap
output = virtualenv.create_bootstrap_script(textwrap.dedent("""
import os, subprocess, sys
def after_install(options, home_dir):
    import os
    print("Home dir: ", home_dir)
    print("CWD: ", os.getcwd())
    if sys.platform == 'win32':
        bin = 'Scripts'
        try:
            os.environ['PATH'] = os.environ['JENKINS-PYX64'] + ';' + os.environ['PATH']
        except KeyError:
            pass
    else:
        bin = 'bin'
    pip = os.path.abspath(os.path.join(home_dir, bin, 'pip'))
    subprocess.call([pip, 'install', 'natcap.versioner', '-I'], env=os.environ)
    subprocess.call(
        [pip, 'install',
         'git+https://github.com/phargogh/paver@natcap-version',
         'git+https://github.com/pyinstaller/pyinstaller.git@v3.2.1',
         'pandas',
         'numexpr>=2.4.6',
         'numpy',
         'scipy',
         'cvxpy==0.4.8',
         'natcap.invest==3.3.3',
         'pygeoprocessing==0.3.3',
        ], env=os.environ)
"""))
print(output)
