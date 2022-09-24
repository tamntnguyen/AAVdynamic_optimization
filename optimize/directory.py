import os
import sys


def detect_root(path):
    try:
        # folderlist = [d for d in os.listdir(path) if os.path.isdir(d)]
        folderlist = [d for d in os.listdir(path)]
        if 'script' in folderlist:
            return True
        else:
            return False
    except:
        return False
        
def find_root(path):
    max_depth = 5
    depth = 0
    while 1:
        path = os.path.abspath(os.path.join(path, os.pardir))        
        if detect_root(path):
            break
        elif depth > max_depth:
            raise('Cannot find root.')
        depth += 1
    return path


def set_path(workingdir):
    rootdir = find_root(workingdir)
    datadir = os.path.join(rootdir, 'dat')
    scriptdir = os.path.join(rootdir, 'script')
    sys.path.append(scriptdir)
    return workingdir, rootdir, datadir


if __name__ == '__main__':
    workingdir, rootdir, datadir = set_path(os.path.abspath(__file__))