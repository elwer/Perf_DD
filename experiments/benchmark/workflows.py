import pandas as pd
import numpy as np
np.float = float

import sys
sys.path.append('../../uDD')

from studd.studd_batch import STUDD
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier as RF
#from memory_profiler import memory_usage
from skmultiflow.drift_detection.page_hinkley import PageHinkley as PHT
import scorep
import scorep.user
import time
def Workflow(X, y, delta, window_size, dd_set):
    with scorep.instrumenter.disable("INIT"):
        ucdd = STUDD(X=X, y=y, n_train=window_size)
        ucdd.initial_fit(model=RF(), std_model=RF(), is_iks=True)
        results={}

    if "iks" in dd_set:
        with scorep.instrumenter.disable("INIT"):
            ucdd.initial_fit(model=RF(), std_model=RF(), is_iks=True)
        print("Detecting change with IKS")
        timeA = time.time()
        RES_IKS = ucdd.drift_detection_iks(datastream_=ucdd.datastream,
                                             model_=ucdd.base_model, feature_id=2,
                                             training_data_=ucdd.init_training_data["X"],
                                             n_train_=ucdd.n_train,
                                             n_samples=window_size,
                                             upd_model=True)
        RES_IKS["time"]=time.time()-timeA
        results["IKS"]=RES_IKS

    with scorep.instrumenter.disable("INIT"):
        #fit again for the others, with original student model fitting (iks=False)
        ucdd.initial_fit(model=RF(), std_model=RF(), is_iks=False)

    if "studd" in dd_set:
        print("Detecting change with STUDD")
        timeA = time.time()
        RES_STUDD = ucdd.drift_detection_std(datastream_=ucdd.datastream,
                                            model_=ucdd.base_model,
                                            std_model_=ucdd.student_model,
                                            n_train_=ucdd.n_train,
                                            n_samples=window_size,
                                            delta=delta / 2,
                                            upd_model=True,
                                            upd_std_model=True,
                                            detector=PHT)
        RES_STUDD["time"]=time.time()-timeA
        results["STUDD"]=RES_STUDD

    if "bl1" in dd_set:
        print("Detecting change with bl1")
        timeA = time.time()
        res_bl1 = ucdd.BL1_never_adapt(datastream_=ucdd.datastream,
                                       model_=ucdd.base_model)
        res_bl1["time"]=time.time()-timeA
        results["BL1"]=res_bl1


    if "bl2" in dd_set:
        print("Detecting change with bl2")
        timeA = time.time()
        res_bl2 = ucdd.BL2_retrain_after_w(datastream_=ucdd.datastream,
                                           model_=ucdd.base_model,
                                           n_train_=i,
                                           n_samples=window_size)
        res_bl2["time"]=time.time()-timeA
        results["BL2"]=res_bl2


    perf_acc = dict()
    pointsbought = dict()
    for m in results:
        x = results[m]
        perf_acc_i = metrics.accuracy_score(y_true=x["preds"]["y"],
                                            y_pred=x["preds"]["y_hat"])

        pointsbought[m] = x["samples_used"]
        perf_acc[m] = perf_acc_i

    perf_acc = pd.DataFrame(perf_acc.items())
    perf_acc.columns = ['Method', 'Acc']

    return perf_acc, pointsbought

