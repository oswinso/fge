import pathlib
import sys

import ipdb
from decorator import contextmanager
from og.path_utils import safe_path_exists
from playsound3 import playsound


@contextmanager
def launch_ipdb_on_exception():
    path = pathlib.Path("/home/oswinso/Desktop/fauna-oh-nyo.mp3")
    try:
        yield
    except Exception:
        if safe_path_exists(path):
            playsound(path, block=False)
        e, m, tb = sys.exc_info()
        print(m.__repr__(), file=sys.stderr)
        ipdb.post_mortem(tb)
    finally:
        if safe_path_exists(path):
            playsound(path, block=False)
        pass