import pickle
from workflows import Workflow
import arff
import numpy as np
import pandas as pd
import psutil
from scipy.io import arff
import sys
sys.path.append('../../data')
import getData
import scorep

if __name__ == '__main__':
    args = sys.argv[1:]
    dd_list = [str(arg) for arg in args]
    with scorep.instrumenter.disable("INIT"):
        datasets = getData.getCovtype()
        delta = 0.002

        results = dict()

        data = datasets.get(list(datasets)[0])
        y = data.target.values
        X = data.drop(['target'], axis=1)
        labels=data['target'].unique()

        n_train_obs = 5000
        W = n_train_obs

    predictions, detections, train_size, training_info, results_comp = \
        Workflow(X=X, y=y,delta=delta,window_size=W, dd_set=dd_list)

    ds_results = \
        dict(predictions=predictions,
             detections=detections,
             n_updates=train_size,
             data_size=len(y),
             training_info=training_info,
             results_comp=results_comp)

    results[df] = ds_results

    with open('results/forestcover.pkl', 'wb') as fp:
        pickle.dump(results, fp)
