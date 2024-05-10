import pickle
from workflows import Workflow
import numpy as np
import pandas as pd
import psutil
import sys
sys.path.append('../../data')
import getData

if __name__ == '__main__':
    datasets = getData.getCovtype()
    datasets.update(getData.getAirlines())
    datasets.update(getData.getElec())
    datasets.update(getData.getAbruptInsects())
    datasets.update(getData.getGas())

    results = dict()

    for df in datasets:
        data = datasets.get(df)
        y = data.target.values
        X = data.drop(['target'], axis=1)
        labels=data['target'].unique()
        print("doing " + str(df))
        small_data_streams = ['AbruptInsects',
                              'Gas', 'Electricity']

        if str(df) in small_data_streams:
            # retraining fater n samples
            n = 1
            # sample steps
            s = 5
            # window size retraining
            w = 500
        else:
            # retraining after n samples
            n = 20
            # sample steps
            s = 100
            # window size retraining
            w = 2000


        result = Workflow(X=X, y=y,window_size=w, start_value_n=n, sample_steps=100)
        results[df] = results

        with open('results/baselines.pkl', 'wb') as fp:
            pickle.dump(results, fp)
