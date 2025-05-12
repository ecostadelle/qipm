import numpy as np
import pandas as pd 

from sklearn.utils.parallel import Parallel, delayed

from ._classes import (
    BaseDecisionTree,
    DecisionTreeClassifier,
)

from .qipm import _qipm, _aprf, _ipm
from .dataviz import plot_transition_feat_relevance, join_and_plot
from ._forest import RandomForestClassifier, _get_n_samples_bootstrap
from ._criterion import Criterion
from ._splitter import Splitter

from typing import Union


LIGHTWEIGHT_DATASETS = [
    [ 'Childhood Lead',          'nhanes_lead'             ],
    [ 'FICO HELOC',              'heloc'                   ],
    [ 'Hospital Readmission',    'diabetes_readmission'    ],
    [ 'Voting',                  'anes'                    ],
]

MIDWEIGHT_DATASETS = [
    [ 'College Scorecard',       'college_scorecard'       ],
    [ 'Hypertension',            'brfss_blood_pressure'    ],
]

HEAVYWEIGHT_DATASETS = [
    [ 'ASSISTments',             'assistments'             ],
    [ 'Diabetes',                'brfss_diabetes'          ],
    [ 'Food Stamps',             'acsfoodstamps'           ],
    [ 'ICU Length of Stay',      'mimic_extract_los_3'     ],
    [ 'ICU Mortality',           'mimic_extract_mort_hosp' ],
    [ 'Income',                  'acsincome'               ],
    [ 'Public Health Insurance', 'acspubcov'               ],
    [ 'Sepsis',                  'physionet'               ],
    [ 'Unemployment',            'acsunemployment'         ],
]

AVALIABLE_DATASETS = sorted(
    LIGHTWEIGHT_DATASETS + MIDWEIGHT_DATASETS + HEAVYWEIGHT_DATASETS, 
    key=lambda x: x[0]
)

def ipm(decision_tree:DecisionTreeClassifier, X:np.ndarray): 
    """
    Computes the Importance in Prediction Measure (IPM) for each 
    feature in the dataset based on the given decision tree.

    Parameters
    ----------
    decision_tree : DecisionTreeClassifier
        A trained decision tree object, typically from scikit-learn. The function accesses 
        the underlying tree structure via the `tree_` attribute.

    X : ndarray of shape (n_samples, n_features), dtype=float64
        The input feature matrix for which the IPM is to be computed.

    Returns
    -------
    ipm : ndarray of shape (n_features,)
        An array representing the importance measure of each feature. The importance is 
        computed based on the number of times a feature is encountered across all paths 
        in the decision tree, normalized by the number of samples and the number of features 
        involved in each path.

    Notes
    -----
    - The function works by traversing the decision tree for each sample in `X` and 
      accumulating the feature contributions across the decision paths.
    - The contribution of each feature is computed as the inverse of the number of unique 
      features in the path, divided by the total number of samples.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> from rfbs import ipm
    >>> X = np.array([
    ...     [1.0, 2.0], 
    ...     [3.0, 4.0], 
    ...     [5.0, 6.0], 
    ...     [7.0, 8.0]
    ... ])
    >>> y = np.array([0, 1, 0, 1])
    >>> clf = DecisionTreeClassifier(max_depth=3).fit(X, y)

    The resulting tree structure:
    
    For the input sample X[1] = [3.0, 4.0], the decision path traverses:
      - x[0] (one time),
      - x[1] (one time, even though it appears twice in the tree).

    The expected output, considering unique feature contributions:

    >>> ipm_result = ipm(clf, X[[1]])
    >>> print(ipm_result)
    [0.5, 0.5] 

    """

    return _ipm(decision_tree, X)


def _parallel_qipm(forest: RandomForestClassifier, X_A: np.ndarray, 
                      y_A: np.ndarray, X_B: np.ndarray, metric: int, 
                      normalize: bool, n_jobs: int):
    """
    Parallel computation of QIPM for a RandomForestClassifier.

    Args:
        forest (RandomForestClassifier): Trained RandomForestClassifier.
        X_A (np.ndarray): Input data (in-distribution).
        y_A (np.ndarray): Labels for X_A.
        X_B (np.ndarray): Input data (out-of-distribution).
        metric (int): Metric for weighting (0: accuracy, 1: precision, 2: recall, 3: fmeasure, 4: none).
        normalize (bool): Whether to normalize QIPM values.
        n_jobs (int): Number of parallel jobs.

    Returns:
        np.ndarray: QIPM values for each feature.
    """
    max_samples = _get_n_samples_bootstrap(X_A.shape[0], forest.max_samples)
    result = Parallel(n_jobs=n_jobs, prefer='threads')(
        delayed(_qipm)
        (
            decision_tree,
            X_A,
            y_A,
            X_B,
            metric,
            max_samples,
            normalize,
        ) for decision_tree in forest.estimators_
    )

    result = np.mean(result, axis=0)

    # L1-normalization is necessary because some trees can got qipm equal zero
    result /= result.sum()

    return result


def check_data(data: Union[pd.DataFrame, pd.Series, np.ndarray]) -> np.ndarray:
    if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        return data.values.astype(np.float64)
    elif isinstance(data, np.ndarray):
        return data.astype(np.float64)
    else:
        print(data.dtype)
        raise TypeError('Input data must be a Pandas DataFrame, Series or a NumPy ndarray')


def get_qipm(forest: RandomForestClassifier, 
             X_A: Union[pd.DataFrame, np.ndarray],
             y_A: Union[pd.DataFrame, np.ndarray], 
             X_B: Union[pd.DataFrame, np.ndarray], 
             weighted_by = 'none', 
             normalize = True, n_jobs: int = -1):
    """
    Compute the Quality-weighted Intervention in Prediction Measure (QIPM).

    Args:
        forest (RandomForestClassifier): Trained RandomForestClassifier.
        X_A (np.ndarray): Input data (in-distribution).
        y_A (np.ndarray): Labels for X_A.
        X_B (np.ndarray): Input data (out-of-distribution).
        weighted_by (str, optional): Metric for weighting ('accuracy', 'precision', 'recall', 'fmeasure', or 'none'). Defaults to 'none'.
        normalize (bool, optional): Whether to normalize QIPM values. Defaults to True.
        n_jobs (int, optional): Number of parallel jobs. Defaults to -1.

    Returns:
        np.ndarray: QIPM values for each feature.
    """
    X_A = check_data(X_A)
    y_A = check_data(y_A)
    X_B = check_data(X_B)

    metric = {
        'accuracy': 0, 
        'precision': 1, 
        'recall': 2, 
        'fmeasure': 3, 
        'none': 4
    }

    return _parallel_qipm(forest, X_A, y_A, X_B, metric[weighted_by], 
                          normalize, n_jobs)


def _parallel_ipm(forest: RandomForestClassifier, X: np.ndarray, n_jobs: int):
    """
    Parallel computation of IPM for a RandomForestClassifier.

    Args:
        forest (RandomForestClassifier): Trained RandomForestClassifier.
        X (np.ndarray): Input data.
        n_jobs (int): Number of parallel jobs.

    Returns:
        np.ndarray: IPM values for each feature.
    """
    result = Parallel(n_jobs=n_jobs, prefer='threads')(
        delayed(_ipm)
        (
            decision_tree,
            X,
        ) for decision_tree in forest.estimators_
    )

    result = np.mean(result, axis=0)

    # L1-normalization is necessary because some trees can got qipm equal zero
    result /= result.sum()

    return result


def get_ipm(forest: RandomForestClassifier, 
            X: Union[pd.DataFrame, np.ndarray],
            n_jobs: int = -1):
    """
    Compute the Quality-weighted Intervention in Prediction Measure (QIPM).

    Args:
        forest (RandomForestClassifier): Trained RandomForestClassifier.
        X (np.ndarray): Input data (in-distribution).
        n_jobs (int, optional): Number of parallel jobs. Defaults to -1.

    Returns:
        np.ndarray: IPM values for each feature.
    """
    X = check_data(X)
    return _parallel_ipm(forest, X, n_jobs)


def aprf(cj: np.ndarray, beta: float = 1.0, average_macro: bool = True):
    """
    Compute accuracy, precision, recall, and F-measure (F-beta score) from a multi-class confusion matrix.

    Parameters
    ----------
    cj : np.ndarray, shape (n_classes, n_classes)
        Confusion matrix where rows represent true class labels and columns represent predicted class labels.
    beta : float, optional
        The weight of recall in the F-beta score. Defaults to 1.0 (F1-score).
        A value of `beta=0` calculates precision, and `beta=inf` calculates recall.
    average_macro : bool, optional
        If True, computes macro-averaged precision, recall, and F-measure (averaged equally across all classes).
        If False, computes a weighted average based on the number of true samples in each class.

    Returns
    -------
    tuple of float
        A tuple containing the following metrics in order:
        
        - accuracy (float): The proportion of correctly classified instances.
        - precision (float): The proportion of correctly predicted positive instances.
        - recall (float): The proportion of actual positive instances correctly identified.
        - fmeasure (float): The F-beta score, a weighted harmonic mean of precision and recall.

    Examples
    --------
    >>> import numpy as np
    >>> from rfbs.qipm import aprf
    >>> confusion_matrix = np.array([[30, 2, 1], [4, 50, 3], [2, 5, 60]], dtype=np.intp)
    >>> aprf(confusion_matrix)
    (0.8917, 0.8827, 0.8939, 0.8876)

    Notes
    -----
    - The confusion matrix \(\mathbf{C}\) should have dimensions \((\ell, \ell)\), where \(\ell\) is the number of class labels.
    - Precision and recall are computed either as macro-averaged or weighted, depending on the `average_macro` parameter.
    - The F-measure calculation depends on the value of `beta`, where different values prioritize precision or recall differently.
    """

    return _aprf(cj, beta, average_macro)

__all__ = [
    "BaseDecisionTree",
    "DecisionTreeClassifier",
    "RandomForestClassifier",
    "Criterion",
    "Splitter",
    "AVALIABLE_DATASETS",
    "LIGHTWEIGHT_DATASETS",
    "MIDWEIGHT_DATASETS",
    "HEAVYWEIGHT_DATASETS",
    "get_qipm",
    "aprf",
    "ipm",
    "plot_transition_feat_relevance",
    "join_and_plot"
]