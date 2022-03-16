import os, shutil, stat
from distutils.dir_util import copy_tree

def on_rm_error( func, path, exc_info):
    # path contains the path of the file that couldn't be removed
    # let's just assume that it's read-only and unlink it.
    os.chmod( path, stat.S_IWRITE )
    os.unlink( path )



for dir in os.listdir():
    if dir in ['ES','EN','DE' ]:
        for d in os.listdir(dir):
            if 'LICENSE' in os.listdir(dir + '/' + d):
                full_path = dir + '/' + d + '/LICENSE'
                os.remove(full_path)
            if 'etc' in os.listdir(dir+'/'+d):
                full_path = dir+'/'+d + '/etc'
                shutil.rmtree( full_path, onerror = on_rm_error )



