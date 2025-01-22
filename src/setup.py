import numpy
import sys
from Cython.Build import cythonize
from setuptools import setup, Extension, find_packages

if sys.platform.startswith("win"):
    openmp_arg = '/openmp'
else:
    openmp_arg = '-fopenmp'

extensions = [
    Extension(
        '*',
        ['qipm/*.pyx'],
        define_macros=[(
            'NPY_NO_DEPRECATED_API', 
            'NPY_1_7_API_VERSION'
        )],
        extra_compile_args=[openmp_arg, '-std=c++17'],
        extra_link_args=[openmp_arg],
        language='c++'
    ),
]

setup(
    name='qipm',
    version='0.1',
    ext_modules=cythonize(
        extensions,
        gdb_debug=True,
        language_level='3',
        compiler_directives={
                'boundscheck': False,
                'wraparound': False
        }
    ),
    include_dirs=[
        numpy.get_include(),
    ],
    zip_safe=False,
    packages=find_packages(),
    install_requires=[
        'numpy==1.23.5',
        'scikit-learn==1.5.1',
        'scipy==1.13.1',
        'tableshift @ git+https://github.com/ecostadelle/tableshift.git@necesssary_changes',
        'tabulate',
    ],
    include_package_data=True,
    package_data={'qipm': ['*.pxd', '*.pyx', '*.py']},
)
