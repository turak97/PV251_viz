

import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
from scipy.linalg import toeplitz
from pandas import DataFrame
import numpy as np
import pandas as pd
from scipy.stats import norm


class Model:
    def __init__(self, X, Y):
        self._model = None
        self.refresh(X, Y)  # initializes self._model

    def resid(self):
        return self._model.resid

    def refresh(self, X, Y):
        pass

    def predict(self, min_x, max_x, line_detail):
        pred_x_raw = np.linspace(
            min_x,
            max_x,
            line_detail)
        pred_x = sm.add_constant(pred_x_raw)
        pred_y = self._model.predict(pred_x)
        return pred_x_raw, pred_y

    def _prepare_data_for_fit(self, X, Y):
        X_const = sm.add_constant(X)
        return X_const, Y

    def get_qq_data(self):
        df_res = pd.DataFrame(
            sorted(self._model.resid),
            columns=['residuals']
        )
        res_mean = df_res['residuals'].mean()
        res_std = df_res['residuals'].std()
        df_res['z_actual'] = (df_res['residuals']).map(
            lambda x: (x - res_mean / res_std)
        )
        df_res['rank'] = df_res.index + 1
        df_res['percentile'] = df_res['rank'].map(
            lambda x: x / (len(df_res['residuals']) + 1)
        )
        df_res['z_theoretical'] = norm.ppf(df_res['percentile'])
        return df_res['z_theoretical'], df_res['z_actual']


class OLS_model(Model):
    def __init__(self, X, Y):
        Model.__init__(self, X, Y)

    def create_model_and_fit(self, X, Y):
        return sm.OLS(Y, X).fit()

    def refresh(self, X, Y):
        X, Y = self._prepare_data_for_fit(X, Y)
        self._model = sm.OLS(Y, X).fit()


class GLS_model(Model):
    def __init__(self, X, Y):
        Model.__init__(self, X, Y)

    def refresh(self, X, Y):
        X, Y = self._prepare_data_for_fit(X, Y)
        ols_resid = sm.OLS(Y, X).fit().resid  # rezidua OLS
        res_fit = sm.OLS(ols_resid[1:], ols_resid[:-1]).fit()  # vypocet korelace mezi rezidui
        rho = res_fit.params  # autoregresni parametr

        order = toeplitz(np.arange(len(ols_resid)))  # some magic
        sigma = rho ** order

        self._model = sm.GLS(Y, X, sigma=sigma).fit()


class CO_model(Model):
    def __init__(self, X, Y):
        Model.__init__(self, X, Y)

    def _prepare_data_for_fit(self, X, Y):
        return X, Y

    def refresh(self, X, Y):
        X, Y = self._prepare_data_for_fit(X, Y)
        ols_resid = sm.OLS(Y, X).fit().resid  # rezidua OLS
        res_fit = sm.OLS(ols_resid[1:], ols_resid[:-1]).fit()  # vypocet korelace mezi rezidui
        self.theta = res_fit.params[0]  # autoregresni parametr
        # TODO: theta vyclenit nebo neco
        self.theta = 0.7078783  # TODO: !!!!!! DEL !!!!!!!!

        Y_no_autoregression = []
        for (y1, y2) in zip(Y[1:], Y[:-1]):
            y_new = y1 - self.theta * y2
            Y_no_autoregression.append(y_new)

        X_no_autoregression = []
        for (x1, x2) in zip(X[1:], X[:-1]):
            x_new = x1 - self.theta * x2
            X_no_autoregression.append(x_new)

        self._model = sm.OLS(
                Y_no_autoregression,
                sm.add_constant(X_no_autoregression)
            ).fit()

    def predict(self, min_x, max_x, line_detail):
        pred_x_raw = np.linspace(
            min_x,
            max_x,
            line_detail)
        [beta_0, beta_1] = self._model.params
        beta_0 = beta_0 / (1 - self.theta)

        pred_y = []
        for x in pred_x_raw:
            pred_y.append(x * beta_1 + beta_0)
        return pred_x_raw, pred_y

