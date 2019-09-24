import numpy as np

def max_rule(classifiers_list, csf_axis=0):
    final = np.amax(classifiers_list, axis=csf_axis)
    return np.argmax(final, axis=1), final

def sum_rule(classifiers_list, csf_axis=0):
    final = np.sum(classifiers_list, axis=csf_axis)
    return np.argmax(final, axis=1), final

def product_rule(classifiers_list, csf_axis=0):
    final = np.prod(classifiers_list, axis=csf_axis)
    return np.argmax(final, axis=1), final

# ---------------------------------------------------------
def min_rule(classifiers_list, csf_axis=0):
    final = np.amin(classifiers_list, axis=csf_axis)
    return np.argmax(final, axis=1), final

def median_rule(classifiers_list, csf_axis=0):
    final = np.median(classifiers_list, axis=csf_axis)
    return np.argmax(final,axis=1), final
