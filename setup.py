#!/usr/bin/env python

from distutils.core import setup
from distutils.extension import Extension
import numpy as np
import os.path

cython_req = (0, 16)
try:
    import Cython
    from Cython.Distutils import build_ext
    import re
    cython_ok = tuple(map(int, re.sub(r"[^\d.]*", "", Cython.__version__).split('.')[:2])) >= cython_req 
except ImportError:
    cython_ok = False 

if not cython_ok:
    raise ImportError("At least Cython {0} is required".format(".".join(map(str, cython_req))))

def cython_extension(modpath, mp=False):
    extra_compile_args = []
    extra_link_args = []
    if mp:
        extra_compile_args.append('-fopenmp')
        extra_link_args.append('-fopenmp') 
    filepath = os.path.join(*modpath.split('.')) + ".pyx"
    return Extension(modpath, [filepath], extra_compile_args=extra_compile_args, extra_link_args=extra_link_args)


setup(name='shiftem',
    cmdclass = {'build_ext': build_ext},
    version='0',
    url="https://github.com/markstoehr/shiftem",
    description="Latent Shift EM algorithm for product of bernoulli moels",
    packages=[
        'src', 'local'
    ],
    ext_modules = [
        cython_extension("src.bernoullishiftonly_em_fast"),

    ], 
)
