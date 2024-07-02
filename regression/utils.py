from scipy.stats import pearsonr

def calculate_pearson(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]
