cimport numpy as cnp

ctypedef   Py_ssize_t intp_t
ctypedef       double float64_t
ctypedef   signed int int32_t
ctypedef unsigned int uint32_t

# Metric indices for QIPM computation.
cdef:
    intp_t ACCURACY
    intp_t PRECISION
    intp_t RECALL
    intp_t FMEASURE

cdef intp_t rand_pdf(
    uint32_t* random_state, 
    intp_t low, 
    intp_t high, 
    intp_t[::1] features, 
    const float64_t[:] feature_bias
) noexcept nogil

cpdef cnp.ndarray _ipm(object decision_tree, float64_t[:, :] X)

cdef cnp.ndarray get_confusion_matrices(object decision_tree, float64_t[:,:] X, float64_t[:] y, intp_t[:] index)

cdef void cj_to_mcm(intp_t[:,:] cj, intp_t* mcm) noexcept nogil

cdef void _prf_divide(float64_t* num, float64_t* den, float64_t* res, intp_t n) noexcept nogil

cdef float64_t accuracy_from_confusion_matrix(intp_t[:, :] cj) noexcept nogil

cdef float64_t precision_from_confusion_matrix(intp_t[:,:] cj, bint average_macro=*) noexcept nogil

cdef float64_t recall_from_confusion_matrix(intp_t[:,:] cj, bint average_macro=*) noexcept nogil

cdef float64_t fmeasure_from_confusion_matrix(intp_t[:,:] cj, float64_t beta=*, bint average_macro=*) noexcept nogil

cpdef cnp.ndarray _qipm(object decision_tree, float64_t[:,:] X_A, float64_t[:] y_A, float64_t[:,:] X_B, intp_t metric, intp_t max_samples, bint normalize=*)