from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# Explicitly define the extension name as a flat module
ext = Extension(
    name="quadtree_core", 
    sources=["quadtree_core.pyx"],
    include_dirs=[np.get_include()]
)

setup(
    ext_modules=cythonize(ext, compiler_directives={'language_level': "3"})
)