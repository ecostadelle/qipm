import numpy
import sys
from Cython.Build import cythonize
from setuptools import setup, Extension, find_packages
import os

if sys.platform.startswith("win"):
    openmp_arg = '/openmp'
else:
    openmp_arg = '-fopenmp'

long_description = """
# Quality-weighted Intervention in Prediction Measure (QIPM)

**QIPM** is an innovative instance-based feature importance metric for the Random Forest (RF) classifier. The primary objective of QIPM is to provide a more accurate and interpretable assessment of feature importance by incorporating quality metrics such as **accuracy** and **F1-score**, enhancing the predictive capabilities of machine learning models in **dataset shift** scenarios.

## **Context**
Supervised machine learning models often assume that training and deployment datasets share the same underlying distribution. However, this assumption rarely holds true in real-world applications, leading to **dataset shift**, a discrepancy that can significantly degrade model performance. QIPM was developed to address this challenge by offering a refined approach to feature importance estimation.

## **Methodology**
The proposed approach combines two key elements:

1. **Refined Feature Importance Estimation:**  
   - QIPM builds upon the **Intervention in Prediction Measure (IPM)** by extending it to incorporate quality metrics derived from feature-specific confusion matrices.  
   - The quality metrics are computed based on labeled data from the source (in-distribution) dataset and are subsequently generalized to the target (out-of-distribution) dataset.

2. **Adapted Decision Tree Construction:**  
   - The method adapts the **Biased Splitting (BS)** technique to prioritize feature selection based on their importance under the new data distribution.  
   - This is achieved by applying the QIPM vector during decision tree construction, ensuring that the most relevant features for the shifted data distribution are given higher priority.

## **Results**
Experimental evaluations demonstrate that applying QIPM in a Random Forest with Biased Splitting (RFBS) significantly improves performance on 13 out of 15 benchmark datasets, as measured by accuracy and F1-score metrics. Compared to the baseline model, QIPM achieved:

- **A statistically significant accuracy improvement (p-value = 0.010)**
- **Macro-averaged F1-score improvements in 12 out of 15 datasets (p-value = 0.026)**

## **Conclusion**
The results suggest that the QIPM-based approach is effective in enhancing the interpretability and performance of Random Forest models in dataset shift scenarios. The proposed solution provides a balanced trade-off between **interpretability and performance**, positioning itself as a promising alternative compared to deep learning models.
"""

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
    author='Ewerton Costadelle',
    author_email='ecostadelle@id.uff.br',
    description='QIPM is a feature importance metric for Random Forest models',
    long_description=long_description,
    long_description_content_type='text/markdown', 
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
    package_data={'qipm': ['*.pxd', '*.pyx', '*.py'] },
)