from cython.cimports.libc cimport math
import numpy as np

from libcpp.unordered_set cimport unordered_set
from libc.stdlib cimport malloc, free

from ._tree cimport Tree
from ._tree cimport Node
from ._utils cimport rand_int, rand_uniform
from ._forest import RandomForestClassifier

from sklearn.ensemble._forest import _generate_sample_indices, _get_n_samples_bootstrap

cdef inline intp_t rand_pdf(uint32_t* random_state, intp_t low, 
                            intp_t high, intp_t[::1] features,
                            const float64_t[:] feature_bias) noexcept nogil:
    """
    Selects a random index from the range [low, high) with probabilities 
    proportional to the values in feature_bias. If all values are zero, 
    a random index is chosen uniformly.
    
    Parameters:
    -----------
    random_state : uint32_t*
        Pointer to the random state for reproducibility.
    low : intp_t
        Lower bound of the index range (inclusive).
    high : intp_t
        Upper bound of the index range (exclusive).
    features : intp_t[:]
        Array of feature indices.
    feature_bias : float64_t[:]
        Array of bias values corresponding to features.

    Returns:
    --------
    intp_t
        Selected index based on the bias distribution.
    """

    cdef:
        float64_t random_value, cumulative_sum = 0.0
        intp_t i, first_nonzero_prob = 0, K = high - low

    # Handle cases where K is not valid (K < 1 means no elements)
    if K < 1:
        # Emit a error if K is invalid
        with gil: 
            raise ValueError(
                "Invalid range: the number of features must be >= 1"
                )
    # If there's only one element, return the first index (0)
    if K == 1:
        return 0

    # Compute the cumulative sum of feature bias ($\sum_{i=0}^{K-1}p_i$) 
    # and track the first nonzero element
    for i in range(K):
        if feature_bias[features[low + i]] < 0:
            with gil: raise ValueError(
                "feature_bias cannot contain negative values"
            )
        
        # The first nonzero probability will be useful if the 
        # random_value is zero.
        if feature_bias[features[low + i]] > 0 and cumulative_sum == 0:
            first_nonzero_prob = i
        
        cumulative_sum += feature_bias[features[low + i]]

    # If all bias values are zero, choose an index uniformly at random
    if cumulative_sum == 0.0:
        return rand_int(low, high, random_state)

    # Generate a random value in the range [0, cumulative_sum]
    random_value = rand_uniform(0, cumulative_sum, random_state)

    # Special case: If the random value is exactly zero, 
    # return the first nonzero probability
    if random_value == 0:
        return first_nonzero_prob

    # Traverse through the cumulative distribution and select the 
    # appropriate index
    cumulative_sum = 0.0
    for i in range(K):
        cumulative_sum += feature_bias[features[low + i]]
        if random_value <= cumulative_sum:
            return i
    # If random_value > cumulative_sum
    with gil: 
        raise ValueError(
            "This case should not be reached under normal conditions"
        )


cpdef cnp.ndarray _ipm(object decision_tree, float64_t[:,:] X): 
    """
    Computes the Importance in Prediction Measure (IPM) for each 
    feature in the dataset based on the given decision tree.

    Parameters
    ----------
    decision_tree : object
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

    cdef:
        Tree tree = <Tree> decision_tree.tree_
        Node* nodes = <Node*> tree.nodes
        unordered_set[intp_t] J
        intp_t i, j, k
        intp_t n = <intp_t> X.shape[0]
        intp_t m = <intp_t> X.shape[1]
        float64_t[:] ipm = np.zeros(m)

    with nogil:
        for i in range(n):
            J.clear()
            k = 0
            while nodes[k].feature > -1:
                j = nodes[k].feature
                J.insert(j)
                if X[i,j] <= nodes[k].threshold:
                    k = nodes[k].left_child
                else:
                    k = nodes[k].right_child
            
            for j in J:
                ipm[j] = ipm[j] + 1/J.size()/n

    return np.asarray(ipm)


cdef cnp.ndarray get_confusion_matrices(object decision_tree, 
                                        float64_t[:,:] X, float64_t[:] y,
                                        intp_t[:] index): 
    """
    Computes confusion matrices for each feature used in decision tree predictions.
    
    Parameters:
    -----------
    decision_tree : object
        A trained decision tree model.
    X : ndarray, shape (n_samples, n_features)
        Feature matrix containing the input samples.
    y : ndarray, shape (n_samples,)
        True class labels corresponding to each sample.
    index : ndarray, shape (n_samples,)
        Indices of samples to be evaluated.

    Returns:
    --------
    ndarray, shape (n_features, n_classes, n_classes)
        A set of confusion matrices for each feature, where the entry [j, u, v] 
        represents the number of instances whose true class is u and predicted class is v.

    Notes:
    ------
    The algorithm iterates through the decision tree for each sample, tracking features
    that contribute to the prediction, and updating the corresponding confusion matrix.
    """

    cdef:
        Tree tree = <Tree> decision_tree.tree_
        Node* nodes = <Node*> tree.nodes
        unordered_set[intp_t] J
        intp_t i, j, k, l, u, v
        intp_t n = <intp_t> index.shape[0]
        intp_t m = <intp_t> X.shape[1]
        # ell is the number of the class labels
        intp_t ell = len(np.unique(y))
        # C encapsulates all m confusion matrices
        intp_t[:,:,:] C = np.zeros((m, ell, ell), dtype=np.intp)

    with nogil:
        for i in index:
            # Clear the contributing features set
            # J is a set of useful feats in current instance
            J.clear()

            # define k as index for current tree node
            k = 0

            # Traverse the tree for this instance
            while nodes[k].feature > -1:
                j = nodes[k].feature
                J.insert(j)
                if X[i,j] <= nodes[k].threshold:
                    k = nodes[k].left_child
                else:
                    k = nodes[k].right_child

            # True class
            u = <intp_t> y[i]

            # Predicted class (highest value at the leaf node)
            v = 0
            for l in range(ell):
                if tree.value[k*2+l] > tree.value[k*2+v]:
                    v = l

            # Increment confusion matrix entries 
            # for each contributing feature
            for j in J:
                C[j, u, v] += 1

    # Return the confusion matrices
    return np.asarray(C)


cdef void cj_to_mcm(intp_t[:,:] cj, intp_t* mcm) noexcept nogil:
    """
    Converts a confusion matrix for a feature into multiclass binarized confusion matrices.
    
    Parameters:
    -----------
    cj : ndarray, shape (num_classes, num_classes)
        Confusion matrix for a single feature.
    mcm : ndarray, shape (num_classes, 2, 2)
        Multiclass binarized confusion matrices.
    
    Returns:
    --------
    None
        Updates the MCM array in-place.
    
    Notes:
    ------
    For each class, the function calculates True Positives (TP), False Positives (FP),
    False Negatives (FN), and True Negatives (TN), populating the multiclass binarized
    confusion matrices accordingly.
    """

    cdef:
        intp_t num_classes = cj.shape[0]
        intp_t tp, fp, fn, tn
        intp_t total_sum = 0  # Total sum of the matrix
        intp_t class_idx, non_class_idx

    for class_idx in range(num_classes):
        # True Positives (TP): 
        # Diagonal element for the current class
        tp = cj[class_idx, class_idx]

        # Initialize FP and FN
        fp = 0
        fn = 0

        for non_class_idx in range(num_classes):
            total_sum += cj[non_class_idx, class_idx]
            if non_class_idx != class_idx:
                # False Positives (FP): 
                # Sum of other rows in the current column
                fp += cj[non_class_idx, class_idx]

                # False Negatives (FN): 
                # Sum of other columns in the current row
                fn += cj[class_idx, non_class_idx]

        # True Negatives (TN): Total sum minus TP, FP, and FN
        tn = total_sum - (tp + fp + fn)

        # Fill the MCM for the current class
        mcm[class_idx*4 + 0*2 + 0] = tn
        mcm[class_idx*4 + 0*2 + 1] = fp
        mcm[class_idx*4 + 1*2 + 0] = fn
        mcm[class_idx*4 + 1*2 + 1] = tp


cdef void _prf_divide(
    float64_t* num, 
    float64_t* den, 
    float64_t* res, 
    intp_t n
) noexcept nogil:      
    for i in range(n):
        res[i] = num[i] / den[i] if den[i] > 0 else 0.0


cdef float64_t accuracy_from_confusion_matrix(intp_t[:,:] cj) noexcept nogil:
    """
    Compute the accuracy from a multi-class confusion matrix.

    Parameters
    ----------
    cj : intp_t[:, :], shape (n_classes, n_classes)
        Confusion matrix where rows represent true class labels and columns represent predicted class labels.

    Returns
    -------
    float64_t
        The computed accuracy, which represents the proportion of correctly classified instances.

    Examples
    --------
    >>> import numpy as np
    >>> from rfbs.qipm cimport accuracy_from_confusion_matrix
    >>> confusion_matrix = np.array([[30, 2, 1], [4, 50, 3], [2, 5, 60]], dtype=np.intp)
    >>> accuracy_from_confusion_matrix(confusion_matrix)
    0.8917
    """

    cdef:
        intp_t i, n_classes = cj.shape[0]
        intp_t cj_sum=0, cj_trace=0
        float64_t accuracy

    for i in range(n_classes):
        for j in range(n_classes):
            cj_sum += cj[i,j]
            if i == j:
                # Cumulative sum along diagonals of the array C^j
                # `trace` because it is same function np.trace()
                cj_trace += cj[i,j]

    accuracy = cj_trace / cj_sum if cj_sum > 0 else 0.0

    return accuracy


cdef float64_t precision_from_confusion_matrix(intp_t[:,:] cj, 
                                               bint average_macro=True
                                               ) noexcept nogil:
    """
    Compute the precision from a multi-class confusion matrix.

    Parameters
    ----------
    cj : intp_t[:, :], shape (n_classes, n_classes)
        Confusion matrix where rows represent true class labels and columns represent predicted class labels.
    average_macro : bool, optional
        If True, computes the macro-averaged precision (averaged equally across all classes).
        If False, computes a weighted precision based on the number of true samples in each class.

    Returns
    -------
    float64_t
        The computed precision, which measures the proportion of correctly predicted positive instances.

    Examples
    --------
    >>> import numpy as np
    >>> from rfbs.qipm cimport precision_from_confusion_matrix
    >>> confusion_matrix = np.array([[30, 2, 1], [4, 50, 3], [2, 5, 60]], dtype=np.intp)
    >>> precision_from_confusion_matrix(confusion_matrix)
    0.8827
    """

    cdef:
        intp_t i, j, n_classes = cj.shape[0]
        intp_t vector_size = n_classes * sizeof(float64_t)
        float64_t tp_sum=0, pred_sum=0, true_sum=0, precision_avg=0.0

        # I need to remember to free them
        float64_t* tp = <float64_t *> malloc(vector_size)
        float64_t* pred = <float64_t *> malloc(vector_size)
        float64_t* true = <float64_t *> malloc(vector_size)
        float64_t* weights = <float64_t *> malloc(vector_size)
        float64_t* precision = <float64_t *> malloc(vector_size)
        intp_t* MCM =  <intp_t *> malloc(n_classes * 2 * 2 * sizeof(intp_t))

    cj_to_mcm(cj, MCM)
    
    # Calculate tp_sum, pred_sum, true_sum ###
    for i in range(n_classes):
        tp[i] = MCM[i*4 + 1*2 + 1]
        pred[i] = tp[i] + (MCM[i*4 + 0*2 + 1])
        true[i] = tp[i] + (MCM[i*4 + 1*2 + 0])
        true_sum += true[i]
    
    # zero_division:
    # Divide, and on zero-division, set scores and/or warn according to
    _prf_divide(tp, pred, precision, n_classes)

    # Average the results
    for i in range(n_classes):
        if average_macro:
            weights[i] = 1 / n_classes
        else: # "weighted"
            weights[i] = true[i] / true_sum

    for i in range(n_classes):
            precision_avg += precision[i] * weights[i]

    free(tp)
    free(pred)
    free(true)
    free(weights)
    free(precision)
    free(MCM)

    return precision_avg


cdef float64_t recall_from_confusion_matrix(intp_t[:,:] cj,
                                            bint average_macro=True
                                            ) noexcept nogil:
    """
    Compute the recall from a multi-class confusion matrix.

    Parameters
    ----------
    cj : intp_t[:, :], shape (n_classes, n_classes)
        Confusion matrix where rows represent true class labels and columns represent predicted class labels.
    average_macro : bool, optional
        If True, computes the macro-averaged recall (averaged equally across all classes).
        If False, computes a weighted recall based on the number of true samples in each class.

    Returns
    -------
    float64_t
        The computed recall, which measures the proportion of actual positive instances correctly identified.

    Examples
    --------
    >>> import numpy as np
    >>> from rfbs.qipm cimport recall_from_confusion_matrix
    >>> confusion_matrix = np.array([[30, 2, 1], [4, 50, 3], [2, 5, 60]], dtype=np.intp)
    >>> recall_from_confusion_matrix(confusion_matrix)
    0.8939
    """

    cdef:
        intp_t i, j, n_classes = cj.shape[0]
        intp_t vector_size = n_classes * sizeof(float64_t)
        float64_t tp_sum=0, pred_sum=0, true_sum=0, recall_avg=0.0

        # I need to remember to free them
        float64_t* tp = <float64_t *> malloc(vector_size)
        float64_t* pred = <float64_t *> malloc(vector_size)
        float64_t* true = <float64_t *> malloc(vector_size)
        float64_t* weights = <float64_t *> malloc(vector_size)
        float64_t* recall = <float64_t *> malloc(vector_size)
        intp_t* MCM =  <intp_t *> malloc(n_classes * 2 * 2 * sizeof(intp_t))

    cj_to_mcm(cj, MCM)
    
    # Calculate tp_sum, pred_sum, true_sum ###
    for i in range(n_classes):
        tp[i] = MCM[i*4 + 1*2 + 1]
        pred[i] = tp[i] + (MCM[i*4 + 0*2 + 1])
        true[i] = tp[i] + (MCM[i*4 + 1*2 + 0])
        true_sum += true[i]
    
    # zero_division:
    # Divide, and on zero-division, set scores and/or warn according to
    _prf_divide(tp, true, recall, n_classes)

    # Average the results
    for i in range(n_classes):
        if average_macro:
            weights[i] = 1 / n_classes
        else: # "weighted"
            weights[i] = true[i] / true_sum

    for i in range(n_classes):
        recall_avg += recall[i] * weights[i]

    free(tp)
    free(pred)
    free(true)
    free(weights)
    free(recall)
    free(MCM)

    return recall_avg


cdef float64_t fmeasure_from_confusion_matrix(intp_t[:,:] cj,
                                              float64_t beta=1.0,
                                              bint average_macro=True
                                              ) noexcept nogil:
    """
    Compute the F-measure (F-beta score) from a multi-class confusion matrix.

    Parameters
    ----------
    cj : intp_t[:, :], shape (n_classes, n_classes)
        Confusion matrix where rows represent true class labels and columns represent predicted class labels.
    beta : float, optional
        The weight of recall in the F-beta score. Defaults to 1.0 (F1-score).
    average_macro : bool, optional
        If True, computes the macro-averaged F-measure (averaged equally across all classes).
        If False, computes a weighted F-measure based on the number of true samples in each class.

    Returns
    -------
    float64_t
        The computed F-measure (F-beta score).

    Examples
    --------
    >>> import numpy as np
    >>> from rfbs.qipm cimport fmeasure_from_confusion_matrix
    >>> confusion_matrix = np.array([[30, 2, 1], [4, 50, 3], [2, 5, 60]], dtype=np.intp)
    >>> fmeasure_from_confusion_matrix(confusion_matrix)
    0.8876
    """

    cdef:
        intp_t i, j, n_classes = cj.shape[0]
        intp_t vector_size = n_classes * sizeof(float64_t)
        float64_t tp_sum=0, pred_sum=0, true_sum=0, beta2=beta**2
        float64_t fmeasure_avg=0.0

        # I need to remember to free them
        float64_t* tp = <float64_t *> malloc(vector_size)
        float64_t* pred = <float64_t *> malloc(vector_size)
        float64_t* tp_beta = <float64_t *> malloc(vector_size)
        float64_t* true = <float64_t *> malloc(vector_size)
        float64_t* weights = <float64_t *> malloc(vector_size)
        float64_t* denom = <float64_t *> malloc(vector_size)
        float64_t* fmeasure = <float64_t *> malloc(vector_size)
        intp_t* MCM =  <intp_t *> malloc(n_classes * 2 * 2 * sizeof(intp_t))

    cj_to_mcm(cj, MCM)

    # Calculate tp_sum, pred_sum, true_sum ###
    for i in range(n_classes):
        tp[i] = MCM[i*4 + 1*2 + 1]
        pred[i] = tp[i] + (MCM[i*4 + 0*2 + 1])
        true[i] = tp[i] + (MCM[i*4 + 1*2 + 0])
        true_sum += true[i]

        if math.isfinite(beta) and beta > 0:
            denom[i] = beta2 * true[i] + pred[i]
            tp_beta[i] = (1 + beta2) * tp[i]

    if math.isinf(beta):
        # fmeasure = recall
        _prf_divide(tp, true, fmeasure, n_classes)
    elif beta == 0:
        # fmeasure = precision
        _prf_divide(tp, pred, fmeasure, n_classes)
    else:
        # The score is defined as:
        # score = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
        # Therefore, we can express the score in terms of confusion matrix entries as:
        # score = (1 + beta**2) * tp / ((1 + beta**2) * tp + beta**2 * fn + fp)
        _prf_divide(tp_beta, denom, fmeasure, n_classes)

    # Average the results
    for i in range(n_classes):
        if average_macro:
            weights[i] = 1 / n_classes
        else: # "weighted":
            weights[i] = true[i] / true_sum

    for i in range(n_classes):
        fmeasure_avg += fmeasure[i] * weights[i]

    free(tp)
    free(pred)
    free(tp_beta)
    free(true)
    free(weights)
    free(denom)
    free(fmeasure)
    free(MCM)

    return fmeasure_avg


cpdef cnp.ndarray _qipm(object decision_tree, float64_t[:,:] X_A, 
                        float64_t[:] y_A, float64_t[:,:] X_B, 
                        intp_t metric, intp_t max_samples, 
                        bint normalize=True):
    """
    Compute QIPM for a single decision tree.

    Parameters:
    -----------
    decision_tree : object
        Decision tree object.
    X_A : ndarray, shape (n_samples_A, n_features)
        Input data (in-distribution).
    y_A : ndarray, shape (n_samples_A,)
        Labels for X_A.
    X_B : ndarray, shape (n_samples_B, n_features)
        Input data (out-of-distribution).
    metric : int
        Metric for weighting (0: accuracy, 1: precision, 2: recall, 3: fmeasure, 4: none).
    max_samples : int
        Maximum samples used for bootstrapping.
    normalize : bool, optional (default=True)
        If True it normalizes QIPM values at each tree.

    Returns:
    --------
    ndarray
        QIPM values for each feature.

    Notes:
    ------
    This function calculates the Quality-weighted Importance in Prediction Measure (QIPM) 
    for a single decision tree by leveraging confusion matrices and weighting metrics. 
    If normalization is enabled, the computed values are scaled at each tree.
    """

    cdef:
        int32_t random_state = decision_tree.random_state
        intp_t i, j, k, l, u, v
        intp_t n = <intp_t> X_A.shape[0]
        intp_t m = <intp_t> X_A.shape[1]
        # ell is the number of the class labels
        intp_t ell = len(np.unique(y_A))
        # C encapsulates all m confusion matrices
        intp_t[:,:,:] C = np.zeros((m, ell, ell), dtype=np.intp)
        float64_t normalizer = 0.0
        float64_t[:] qipm = np.zeros(m, dtype=np.float64)
        float64_t multiplier, class_

    # oob means out-of-bag, it refers to base A
    oob_idx = get_oob_idx(decision_tree, n, max_samples)
    C = get_confusion_matrices(decision_tree, X_A, y_A, oob_idx)
    # get original ipm for and quality weighting
    qipm = _ipm(decision_tree, X_B)
    with nogil:
        for j in range(m):

            if metric == ACCURACY:
                multiplier = accuracy_from_confusion_matrix(C[j,:,:])
            elif metric == PRECISION:
                multiplier = precision_from_confusion_matrix(C[j,:,:])
            elif metric == RECALL:
                multiplier = recall_from_confusion_matrix(C[j,:,:])
            elif metric == FMEASURE:
                multiplier = fmeasure_from_confusion_matrix(C[j,:,:])
            else:
                multiplier = 1.0

            qipm[j] *= multiplier
            normalizer += qipm[j]

        if normalize and (normalizer > 0):
            for j in range(m):
                qipm[j] /= normalizer
    
    return np.asarray(qipm)


def get_oob_idx(decision_tree, n, max_samples):
    """
    Retrieve out-of-bag (OOB) indices for a given decision tree.

    Parameters:
    -----------
    decision_tree : object
        Decision tree object.
    n : int
        Total number of samples.
    max_samples : float or int
        The maximum number of samples for bootstrapping.

    Returns:
    --------
    ndarray
        Indices of OOB samples.
    """

    random_state = decision_tree.random_state
    bag_idx = _generate_sample_indices(random_state, n, max_samples)
    sample_counts = np.bincount(bag_idx, minlength=n)
    oob_idx = np.arange(n, dtype=np.intp)[sample_counts==0]

    return oob_idx

def _aprf(cj: np.ndarray, beta: float = 1.0, average_macro: bool = True):
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

    return (
        accuracy_from_confusion_matrix(cj), 
        precision_from_confusion_matrix(cj,average_macro), 
        recall_from_confusion_matrix(cj,average_macro), 
        fmeasure_from_confusion_matrix(cj,beta,average_macro),
    )
