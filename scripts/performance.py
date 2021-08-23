from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Plot CM
def plot_cm(cm,i, loss_function, name_metric, ml_model):
    df_cm = pd.DataFrame(cm)
    df_cm.index.name = 'Actual class'
    df_cm.columns.name = 'Predicted class'
    plt.figure(figsize = (10,7))
    plt.title("Model F")
    sns.set(font_scale=1.4)
    sns.heatmap(df_cm, cmap="Blues", annot=True, fmt='d', annot_kws={"size":16})
    tn, fp, fn, tp = cm.ravel()
    plt.savefig("cm/{}_cm_{}_{}_{}.png".format(ml_model,i,loss_function,name_metric))
    print("True negative: ", tn,"False Positive: ", fp, "False Negative:", fn,"True Positive: ", tp)

def threshold_metrics(y_test, y_pred_test, loss_function, name_metric, ml_model):
    testing_list =[]
    for i in  np.linspace(0.1,0.9,9):
        d = {}
        y_p = y_pred_test >= i
        print("i",i, y_test.shape, y_pred_test.shape)
        cm = confusion_matrix(y_test,y_p)
        np.save("cm/{}_cm_{}_{}_{}.npy".format(ml_model,i,loss_function,name_metric),cm)
        plot_cm(cm,i,loss_function, name_metric,ml_model)
        d["th"] = i
        d["F1"] = np.round(f1_score(y_test,y_p),2)
        d["Prec"] = np.round(precision_score(y_test,y_p),2)
        d["Rec"] = np.round(recall_score(y_test,y_p),2)
        d["Acc"] = np.round(accuracy_score(y_test,y_p),2)
        d["RocAuc"] = np.round(roc_auc_score(y_test,y_p),2)
        testing_list.append(d)
    return testing_list
