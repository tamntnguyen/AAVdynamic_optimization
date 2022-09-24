# Automatic installtion
#
# 1. Install Anaconda3 and create an environment
# 2. Select the environment as a default interpreter of your IDE (vscode, pycharm, etc.)
# 3. Run the code below

import sys
import subprocess
import pkg_resources
required = {'numpy', 'pyomo', 'idaes-pse', 'termcolor', 'hdf5storage', 'dill'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed

def run_command(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
    rc = process.poll()
    return rc


if missing:
    python = sys.executable
    run_command([python, '-m', 'pip', 'install', *missing])
    run_command(['idaes', 'get-extensions'])
    
