import re
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score, \
    f1_score

def evaluate_b(y, y_pred):
    report_text = classification_report(y, y_pred, target_names=['nsbr', 'sbr'])
    # print(report_text)
    report_list = re.sub(r'[\n\s]{1,}', ' ', report_text).strip().split(' ')
    # print('Confusion Matrix:')
    conf_matrix = confusion_matrix(y, y_pred)
    # print(conf_matrix)
    TN = conf_matrix.item((0, 0))
    FN = conf_matrix.item((1, 0))
    TP = conf_matrix.item((1, 1))
    FP = conf_matrix.item((0, 1))
    pd = 100 * TP / (TP + FN)
    pf = 100 * FP / (FP + TN)
    g_measure = 2 * pd * (100 - pf) / (pd + 100 - pf)
    # print('g-measure:%s' % g_measure)
    # print('precision:%s' % precision_score(y, y_pred, average='binary'))
    # print('recall:%s' % recall_score(y, y_pred, average='binary'))
    # print('f-measure:%s' % f1_score(y, y_pred, average='binary'))
    prec = 100 * precision_score(y, y_pred, average='binary')
    recall = 100 * recall_score(y, y_pred, average='binary')
    f_measure = 100 * f1_score(y, y_pred, average='binary')

    accuracy = 100 * accuracy_score(y, y_pred)
    result = [TN, TP, FN, FP, pd, pf, prec, f_measure, g_measure]
    # print(result)
    return result