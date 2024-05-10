import scorep
import scorep.user
from skmultiflow.data.data_stream import DataStream
from skmultiflow.drift_detection.page_hinkley import PageHinkley as PHT
import copy
import numpy as np
import time
#from memory_profiler import memory_usage, profile
from scipy.stats import ks_2samp
from iks.IKS import IKS
from random import random

class STUDD:

    def __init__(self, X, y, n_train):
        """

        :param X:
        :param y:
        :param n_train:
        """

        D = DataStream(X, y)
        D.prepare_for_use()

        self.datastream = D
        self.n_train = n_train
        self.W = n_train
        self.base_model = None
        self.student_model = None
        self.init_training_data = None

    def initial_fit(self, model, std_model, is_iks):
        """

        :return:
        """

        X_tr, y_tr = self.datastream.next_sample(self.n_train)

        model.fit(X_tr, y_tr)

        yhat_tr = model.predict(X_tr)
        if not is_iks:
            with scorep.user.region("STUDD_DriftDetection"):
                std_model.fit(X_tr, yhat_tr)
            self.student_model = std_model
        else:
            self.student_model = None
        self.base_model = model
        self.init_training_data = dict({"X": X_tr, "y": y_tr, "y_hat": yhat_tr})


    DETECTOR = PHT


    @staticmethod
    def drift_detection_std(datastream_, model_,
                            std_model_, n_train_,
                            delta, n_samples,
                            upd_model=False,
                            upd_std_model=True,
                            detector=DETECTOR):
        with scorep.instrumenter.disable("INIT"):
            datastream = copy.deepcopy(datastream_)
            base_model = copy.deepcopy(model_)
            n_train = copy.deepcopy(n_train_)
            std_alarms = []
            iter = n_train
            n_updates = 0
            samples_used = 0
            y_hat_hist = []
            y_buffer, y_hist = [], []
            X_buffer, X_hist = [], []
            dataUsed = set()
        with scorep.user.region("STUDD_DriftDetection"):
            student_model = copy.deepcopy(std_model_)
            std_detector = detector(delta=delta)
        while datastream.has_more_samples():
            with scorep.instrumenter.disable("MAINTAIN_BUFFERS"):
                Xi, yi = datastream.next_sample()
                y_hist.append(yi[0])
                y_buffer.append(yi[0])
                X_hist.append(Xi[0])
                X_buffer.append(Xi[0])
            with scorep.instrumenter.disable("INFERENCE"):
                model_yhat = base_model.predict(Xi)
                y_hat_hist.append(model_yhat[0])
            with scorep.user.region("STUDD_DriftDetection"):
                std_model_yhat = student_model.predict(Xi)
                std_err = int(model_yhat != std_model_yhat)
                std_detector.add_element(std_err)
                detected_change = std_detector.detected_change()
            if detected_change:
                std_alarms.append(iter)
                if upd_model:
                    with scorep.instrumenter.disable("MAINTAIN_MODEL"):
                        X_buffer = np.array(X_buffer)
                        y_buffer = np.array(y_buffer)
                        samples_used_iter = len(y_buffer[-n_samples:])
                        for i in range(y_buffer.size-500,y_buffer.size):
                            dataUsed.add(i)
                        base_model.fit(X_buffer[-n_samples:],
                                       y_buffer[-n_samples:])
                    with scorep.user.region("STUDD_DriftDetection"):
                        yhat_buffer = base_model.predict(X_buffer)
                        if upd_std_model:
                            student_model.fit(X_buffer, yhat_buffer)
                        else:
                            student_model.fit(X_buffer[-n_samples:],
                                              yhat_buffer[-n_samples:])
                        # y_buffer = []
                        # X_buffer = []
                        y_buffer = list(y_buffer)
                        X_buffer = list(X_buffer)
                        n_updates += 1
                        samples_used += samples_used_iter

            iter += 1

        preds = dict({"y": y_hist, "y_hat": y_hat_hist})

        output = dict({"alarms": std_alarms,
                       "preds": preds,
                       "n_updates": n_updates,
                       "samples_used": samples_used})
        return output
    '''
    @profile
    '''
    @staticmethod
    def drift_detection_iks(datastream_, model_, feature_id,
                            training_data_, n_train_,
                            n_samples,
                            upd_model=False):
        with scorep.instrumenter.disable("INIT"):
            datastream = copy.deepcopy(datastream_)
            base_model = copy.deepcopy(model_)
            n_train = copy.deepcopy(n_train_)
            std_alarms = []
            iter = n_train
            n_updates = 0
            samples_used = 0
            y_hat_hist = []
            y_buffer, y_hist = [], []
            X_buffer, X_hist = [], []
            dataUsed = set()

        with scorep.user.region("IKS_DriftDetection"):
            sliding=[]
            reference=[]
            iks_detector = IKS()
            # fit the iks initially
            for x in training_data_:
                feature0 = (x[feature_id], random())
                feature1 = (x[feature_id], random())
                iks_detector.Add(feature0, 0)
                iks_detector.Add(feature1, 1)
                reference.append(feature0)
                sliding.append(feature1)

        while datastream.has_more_samples():
            # print("Iteration: " + str(iter))
            with scorep.instrumenter.disable("MAINTAIN_BUFFERS"):
                Xi, yi = datastream.next_sample()
                y_hist.append(yi[0])
                y_buffer.append(yi[0])
                X_hist.append(Xi[0])
                X_buffer.append(Xi[0])
            with scorep.instrumenter.disable("INFERENCE"):
                model_yhat = base_model.predict(Xi)
                y_hat_hist.append(model_yhat[0])
            with scorep.user.region("IKS_DriftDetection"):
                iks_detector.Remove(sliding[0], 1)
                sliding.pop(0)
                feature1 = (Xi[0][feature_id], random())
                iks_detector.Add(feature1, 1)
                sliding.append(feature1)
                ca = IKS.CAForPValue(0.00000001)
                detected_change = iks_detector.Test(ca)
            if detected_change:
                #print("Found change std in iter: " + str(iter))
                std_alarms.append(iter)

                #now strategies after detected drift, model replacement, L/B transformation,...
                if upd_model:
                    with scorep.instrumenter.disable("MAINTAIN_MODEL"):
                        #model replacement
                        X_buffer = np.array(X_buffer)
                        y_buffer = np.array(y_buffer)
                        samples_used_iter = len(y_buffer[-n_samples:])
                        for i in range(y_buffer.size-n_samples,y_buffer.size):
                            dataUsed.add(i)

                        base_model.fit(X_buffer[-n_samples:],
                                       y_buffer[-n_samples:])

                    #update iks, exchange windows
                    with scorep.user.region("IKS_DriftDetection"):
                        for x in reference:
                            iks_detector.Remove(x, 0)
                        for y in sliding:
                            iks_detector.Add(y, 0)
                        reference = sliding.copy()
                        y_buffer = list(y_buffer)
                        X_buffer = list(X_buffer)
                        n_updates += 1
                        samples_used += samples_used_iter

            iter += 1



        preds = dict({"y": y_hist, "y_hat": y_hat_hist})
        output = dict({"alarms": std_alarms,
                       "preds": preds,
                       "n_updates": n_updates,
                       "samples_used": samples_used})

        return output

    
    @staticmethod
    def BL2_retrain_after_w(datastream_, model_, n_train_, n_samples):
        import copy
        import numpy as np

        datastream = copy.deepcopy(datastream_)
        model = copy.deepcopy(model_)
        n_train = copy.deepcopy(n_train_)
        iter = copy.deepcopy(n_train_)

        j, n_updates, samples_used = 0, 0, 0
        yhat_hist = []
        y_buffer, y_hist = [], []
        X_buffer, X_hist = [], []
        while datastream.has_more_samples():
            # print("Iteration: " + str(iter))

            Xi, yi = datastream.next_sample()

            y_hist.append(yi[0])
            y_buffer.append(yi[0])
            X_hist.append(Xi[0])
            X_buffer.append(Xi[0])

            model_yhat = model.predict(Xi)

            yhat_hist.append(model_yhat[0])

            if iter % n_train == 0 and iter > n_train + 1:
                X_buffer = np.array(X_buffer)
                y_buffer = np.array(y_buffer)

                samples_used_iter = len(y_buffer[-n_samples:])

                model.fit(X_buffer[-n_samples:],
                          y_buffer[-n_samples:])

                y_buffer = list(y_buffer)
                X_buffer = list(X_buffer)

                n_updates += 1
                samples_used += samples_used_iter

            iter += 1
            j += 1

        preds = dict({"y": y_hist, "y_hat": yhat_hist})

        output = dict({"alarms": [],
                       "preds": preds,
                       "n_updates": n_updates,
                       "samples_used": samples_used})

        return output

    @staticmethod
    def BL1_never_adapt(datastream_, model_):
        import copy

        datastream = copy.deepcopy(datastream_)
        model = copy.deepcopy(model_)

        yhat_hist, y_hist = [], []

        while datastream.has_more_samples():
            # print("Iteration: " + str(iter))

            Xi, yi = datastream.next_sample()

            y_hist.append(yi[0])

            model_yhat = model.predict(Xi)

            yhat_hist.append(model_yhat[0])

        preds = dict({"y": y_hist, "y_hat": yhat_hist})

        output = dict({"alarms": [],
                       "preds": preds,
                       "n_updates": 0,
                       "samples_used": 0})

        return output
