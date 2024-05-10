import pandas as pd
import numpy as np
np.float = float

import sys
sys.path.append('../../uDD')
from studd.studd_batch import STUDD
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier as RF
from skmultiflow.drift_detection.page_hinkley import PageHinkley as PHT
import time

# for each step, initialize a model, then process the stream and measure the time of processing
# then append this time and the achieved accuracy to the results
def Workflow(X, y, window_size, start_value_n, sample_steps):

    for n in range(start_value_n, 100000,sample_steps):
        ucdd = STUDD(X=X, y=y, n_train=window_size)
        ucdd.initial_fit(model=RF(), std_model=RF(), is_iks=False)

        print("Detecting change with bl2 and n=" + str(n))
        timeA = time.time() 
        res_bl2 = ucdd.BL2_retrain_after_w(datastream_=ucdd.datastream,
                                           model_=ucdd.base_model,
                                           n_train_=n,
                                           n_samples=window_size)
        perf_acc_i = metrics.accuracy_score(y_true=res_bl2["preds"]["y"],
                                            y_pred=res_bl2["preds"]["y_hat"])

        results[i] = {"acc": perf_acc_i, "time": time.time()-timeA}

    return results

