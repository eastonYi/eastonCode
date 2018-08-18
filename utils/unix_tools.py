import os
import stat
import shutil

def rm_dir(path):
    if os.path.exists(path):
        for name in os.listdir(path):
            fullname = os.path.join(path,name)
            mode = os.lstat(fullname).st_mode
            if not stat.S_ISDIR(mode):
                os.remove(fullname)


def mk_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def exists(path):
    return os.path.exists(path)


def copy(src, des):
    shutil.copy(src, des)
    return
