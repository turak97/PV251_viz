
import argparse  # to easily parse arguments
import sys  # sys.argv[1:] -- to obtain possible file path

from bokeh.plotting import curdoc


import pandas as pd

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from bokeh.models import PointDrawTool, Paragraph
from bokeh.layouts import row, column

import statsmodels.api as sm
import numpy as np


LINE_DETAIL = 5
PLOT_SIZE = 600
EXTRA_PLOT_SIZE = int(PLOT_SIZE / 3)
CIRCLE_SIZE = 5
DATASET_COLOR = 'gold'
OLS_COLOR = 'mediumblue'
GLS_COLOR = 'limegreen'
CO_COLOR = 'red'

# TODO: refactoring: inicializace malych plotu (spolecnou cast vytvoreni plotu vyclenit)

from statsmodels.tsa.ar_model import AutoReg
from scipy.linalg import toeplitz
from pandas import DataFrame
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
        # # TODO: theta vyclenit nebo neco
        # self.theta = 0.7078783  # TODO: !!!!!! DEL !!!!!!!!

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




class Layout:
    def __init__(self, data_frame, x_name, y_name):
        self._data_source = ColumnDataSource(
            data=dict(
                x=data_frame[x_name],
                y=data_frame[y_name]
            )
        )
        self._source_on_change_func = self._refresh_trigger
        self._data_source.on_change('data', self._source_on_change_func)

        X, Y = self._get_raw_data()
        self._OLS = OLS_model(X, Y)
        self._GLS = GLS_model(X, Y)
        self._CO = CO_model(X, Y)

        self._main_figure = self._init_main_figure(self._data_source, x_name, y_name)

        self._OLS_qq = self._init_qq(self._OLS, OLS_COLOR)
        self._GLS_qq = self._init_qq(self._GLS, GLS_COLOR)
        self._CO_qq = self._init_qq(self._CO, CO_COLOR)

        self._OLS_index = self._init_index_plot(self._OLS, OLS_COLOR)
        self._GLS_index = self._init_index_plot(self._GLS, GLS_COLOR)
        self._CO_index = self._init_index_plot(self._CO, CO_COLOR)

        self._OLS_res_res = self._init_res_res(self._OLS, OLS_COLOR)
        self._GLS_res_res = self._init_res_res(self._GLS, GLS_COLOR)
        self._CO_res_res = self._init_res_res(self._CO, CO_COLOR)

        main_text, quantiles_help, index_help, reziduals_help, plot_help, usage_help, std_data_help \
            = self._get_help_widgets()
        self.layout = column(
            plot_help,
            usage_help,
            std_data_help,
            row(column(self._main_figure, main_text),
                column(self._OLS_qq, self._GLS_qq, self._CO_qq, quantiles_help),
                column(self._OLS_index, self._GLS_index, self._CO_index, index_help, max_width=400),
                column(self._OLS_res_res, self._GLS_res_res, self._CO_res_res, reziduals_help)
            )
        )

    def _get_raw_data(self):
        df = self._data_source.to_df()
        return df['x'].tolist(), df['y'].tolist()

    def _init_main_figure(self, data_source, x_name, y_name):
        """Initialize figure with main data and regression lines"""
        main_figure = figure(match_aspect=True, tools="pan,wheel_zoom,save,reset,box_zoom,lasso_select,box_select",
                             plot_width=PLOT_SIZE, plot_height=PLOT_SIZE, x_axis_label=x_name, y_axis_label=y_name)

        move_circle = main_figure.circle(source=data_source, x='x', y='y',
                                         size=8, fill_color=DATASET_COLOR, line_color='black')

        point_draw_tool = PointDrawTool(renderers=[move_circle], empty_value='black', add=True)
        main_figure.add_tools(point_draw_tool)

        x_ols, y_ols = self._OLS.predict(
            min(self._data_source.data['x']),
            max(self._data_source.data['x']),
            LINE_DETAIL
        )
        main_figure.line(x_ols, y_ols, line_color=OLS_COLOR)  # OLS line is in renderer 1
        x_gls, y_gls = self._OLS.predict(
            min(self._data_source.data['x']),
            max(self._data_source.data['x']),
            LINE_DETAIL
        )
        main_figure.line(x_gls, y_gls, line_color=GLS_COLOR)  # GLS line is in renderer 2
        x_co, y_co = self._CO.predict(
            min(self._data_source.data['x']),
            max(self._data_source.data['x']),
            LINE_DETAIL
        )
        main_figure.line(x_co, y_co, line_color=CO_COLOR)

        return main_figure

    def _get_help_widgets(self):
        models_help = Paragraph(text="""
        When using a linear regression model, you should be aware that the errors should be stochastically independent, 
        and the residuals of the model should be normally distributed. 
        (For more detail check this article: https://en.wikipedia.org/wiki/Errors_and_residuals) 
        This preconditions often breaks something called autocorrelation. 
        That means that the current error in measurement is affected by the previous error. 
        This often happens in time series like measuring the temperature. 
        Fortunately, we have some methods on how to get rid of autocorrelation.	
        We can compute the autocorrelation coefficient as a correlation between residuals[1, ..., n-1] 
        and residuals [2, ..., n]. When this coefficient is close to 1 or -1, we have a problem. 
        One method is GLS, which can take the autocorrelation coefficient into account and do some magic. 
        The other is the Cochranne-Orcutt method, which tries to modify data with this correlation 
        coefficient and eliminate the correlation. For more info you can check these:
        https://en.wikipedia.org/wiki/Autocorrelation
        https://en.wikipedia.org/wiki/Generalized_least_squares
        """)
        quantiles_help = Paragraph(text="""
        The points should be close to the line. 
        """)
        index_help_1 = Paragraph(text="""
        Those should be randomly distributed
        """)
        index_help_2 = Paragraph(text="""
        around Reziduals axe. 
        """)
        index_help = column(index_help_1, index_help_2)
        reziduals_help = Paragraph(text="""
        There should NOT be any correlation apparent. That means the line should be ideally horizontal.
        """)
        plot_help = Paragraph(text="""
        All lines in the main plot represents some form of regression. Blue is Ordinary least squares. 
        Green is Generalized least squares. Red is Cochranne-Orcutt method.
        """)
        usage_help = Paragraph(text="""
        Select Point draw tool in the tool box and click to add point. You can also select one point
        or multiple points by holding shift. Then you can drag them or remove with Backspace.
        You can select more points with lasso. Then select Point draw tool and drag them or remove them.
        """)
        std_data_help = Paragraph(text="""
        Implicit dataset describes number of birds of particular species which were observed on Hawaii.
        For better fit, data on both axes were somehow normalized.
        """)
        return models_help, quantiles_help, index_help, reziduals_help, \
               plot_help, usage_help, std_data_help

    def _init_res_res(self, model, color):
        res_res_plot = figure(plot_width=EXTRA_PLOT_SIZE, plot_height=EXTRA_PLOT_SIZE,
                              x_axis_label="Reziduals r_1, ..., r_n-1", y_axis_label="Reziduals r_2, ..., r_n")
        res_res_plot.yaxis.minor_tick_line_color = None
        res_res_plot.xaxis.minor_tick_line_color = None
        res_res_plot.toolbar_location = None

        resid = model.resid()
        r1 = resid[:-1].tolist()
        r1_c = sm.add_constant(r1)
        r2 = resid[1:].tolist()

        xx, yy = self._instant_fit(
            r1_c,
            r2,
            min(r1),
            max(r1),
        )
        res_res_plot.circle(x=r1, y=r2, size=CIRCLE_SIZE, color=color)
        res_res_plot.line(x=xx, y=yy, color=color)
        return res_res_plot

    def _instant_fit(self, X_const, Y, min_x, max_x):
        pred_x_raw = np.linspace(
            min_x,
            max_x,
            LINE_DETAIL)
        pred_x = sm.add_constant(pred_x_raw)

        pred_y = sm.OLS(Y, X_const).fit().predict(pred_x)
        return pred_x_raw, pred_y

    def _init_index_plot(self, model, color):
        index_plot = figure(plot_width=EXTRA_PLOT_SIZE, plot_height=EXTRA_PLOT_SIZE,
                            x_axis_label="Indexes", y_axis_label="Reziduals")
        index_plot.yaxis.minor_tick_line_color = None
        index_plot.xaxis.minor_tick_line_color = None
        index_plot.toolbar_location = None

        resid = model.resid()
        index_list = [x for x in range(len(resid))]
        index_plot.circle(x=index_list, y=resid, size=CIRCLE_SIZE, color=color)
        index_plot.line(x=index_list, y=resid, color=color)
        return index_plot

    def _init_qq(self, model, color):
        qq_plot = figure(plot_width=EXTRA_PLOT_SIZE, plot_height=EXTRA_PLOT_SIZE,
                         x_axis_label="Theoretical Quantiles", y_axis_label="Sample Quantiles")
        qq_plot.yaxis.minor_tick_line_color = None
        qq_plot.xaxis.minor_tick_line_color = None
        qq_plot.toolbar_location = None

        qq_x, qq_y = model.get_qq_data()
        qq_plot.circle(x=qq_x, y=qq_y, color=color)  # renderer 0

        x_from, x_to, y_from, y_to = self._get_qq_line(qq_x, qq_y)
        qq_plot.line(x=[x_from, x_to], y=[y_from, y_to], color='black')  # renderer 1
        return qq_plot

    def _refresh_trigger(self, attr, old, new):
        self._place_last_added_correctly()
        self._refresh_models()
        self._refresh_figures()
        self._refresh_extra_graphs()

    def _refresh_models(self):
        X, Y = self._get_raw_data()
        self._OLS.refresh(X, Y)
        self._GLS.refresh(X, Y)
        self._CO.refresh(X, Y)

    def _place_last_added_correctly(self):
        self._data_source.remove_on_change('data', self._source_on_change_func)
        X = self._data_source.data['x']
        Y = self._data_source.data['y']

        pairs = list(zip(X, Y))
        pairs.sort(key=lambda x: x[0])
        X, Y = zip(*pairs)  # unzip

        self._data_source.update(
            data=dict(
                x=X,
                y=Y
            )
        )
        self._data_source.on_change('data', self._source_on_change_func)

    def _refresh_extra_graphs(self):
        self._refresh_qq_plot(self._OLS_qq, self._OLS)
        self._refresh_qq_plot(self._GLS_qq, self._GLS)
        self._refresh_qq_plot(self._CO_qq, self._CO)

        self._refresh_index_plot(self._OLS_index, self._GLS)
        self._refresh_index_plot(self._GLS_index, self._GLS)
        self._refresh_index_plot(self._CO_index, self._CO)

        self._refresh_res_res_plot(self._OLS_res_res, self._OLS)
        self._refresh_res_res_plot(self._GLS_res_res, self._GLS)
        self._refresh_res_res_plot(self._CO_res_res, self._CO)

    def _refresh_qq_plot(self, plot, model):
        qq_new_x, qq_new_y = model.get_qq_data()
        plot.renderers[0].data_source.update(
            data=dict(
                x=qq_new_x,
                y=qq_new_y
            )
        )
        x_from, x_to, y_from, y_to = self._get_qq_line(qq_new_x, qq_new_y)
        plot.renderers[1].data_source.update(
            data=dict(
                x=[x_from, x_to],
                y=[y_from, y_to]
            )
        )

    def _get_qq_line(self, qq_x, qq_y):
        # qq_line gained as a line get with 0.25th and 0.75th residual points
        # from two points are calculated line parameters y = p*x + q
        x1, x2 = qq_x[int(len(qq_x) * 0.25)], qq_x[int(len(qq_x) * 0.75)]
        y1, y2 = qq_y[int(len(qq_y) * 0.25)], qq_y[int(len(qq_y) * 0.75)]
        p = (y2 - y1) / (x2 - x1)
        q = y2 - p * x2

        x_from = qq_x[0]
        x_to = qq_x[len(qq_x) - 1]
        y_from = p * x_from + q,
        y_to = p * x_to + q
        return x_from, x_to, y_from, y_to

    def _refresh_index_plot(self, plot, model):
        resid = model.resid()
        plot.renderers[0].data_source.update(
            data=dict(
                x=[x for x in range(len(resid))],
                y=resid
            )
        )
        plot.renderers[1].data_source.update(
            data=dict(
                x=[x for x in range(len(resid))],
                y=resid
            )
        )

    def _refresh_res_res_plot(self, plot, model):
        resid = model.resid()
        r1 = resid[:-1].tolist()
        r2 = resid[1:].tolist()
        plot.renderers[0].data_source.update(
            data=dict(
                x=r1,
                y=r2
            )
        )

        r1_c = sm.add_constant(r1)
        xx, yy = self._instant_fit(
            r1_c,
            r2,
            min(r1),
            max(r1),
        )
        plot.renderers[1].data_source.update(
            data=dict(
                x=xx,
                y=yy
            )
        )

    def _refresh_figures(self):
        x_ols_new, y_ols_new = self._OLS.predict(
            min(self._data_source.data['x']),
            max(self._data_source.data['x']),
            LINE_DETAIL)
        self._main_figure.renderers[1].data_source.update(
            data=dict(
                x=x_ols_new,
                y=y_ols_new
            )
        )

        x_gls_new, y_gls_new = self._GLS.predict(
            min(self._data_source.data['x']),
            max(self._data_source.data['x']),
            LINE_DETAIL)
        self._main_figure.renderers[2].data_source.update(
            data=dict(
                x=x_gls_new,
                y=y_gls_new
            )
        )

        x_co_new, y_co_new = self._CO.predict(
            min(self._data_source.data['x']),
            max(self._data_source.data['x']),
            LINE_DETAIL)
        self._main_figure.renderers[3].data_source.update(
            data=dict(
                x=x_co_new,
                y=y_co_new
            )
        )



def hardwired_data():
    x_name = 'Year, 1956 = 0'
    y_name = 'Birds (sqrt)'
    data_frame = pd.DataFrame()
    data_frame[x_name] = [ 0,  2,  3,  4,  5,  6,  7,  8, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21,
 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
 41, 42, 43, 44, 45, 46, 47]
    data_frame[y_name] = [
 1.4142135623730951, 1.4142135623730951, 3.1622776601683795,
                2.0, 3.1622776601683795, 3.4641016151377544,
 3.1622776601683795, 2.8284271247461903,  4.123105625617661,
 2.6457513110645907,    6.6332495807108, 7.0710678118654755,
 5.0990195135927845, 3.1622776601683795, 2.6457513110645907,
                1.0,  6.164414002968976,  9.591663046625438,
  10.63014581273465,                9.0,  8.888194417315589,
   8.48528137423857,   9.16515138991168,   8.18535277187245,
     6.557438524302,  7.211102550927978,   8.18535277187245,
  7.416198487095663,   6.48074069840786,                6.0,
  5.656854249492381,  9.899494936611665, 10.770329614269007,
 10.344080432788601, 11.357816691600547, 15.491933384829668,
 17.146428199482248,  18.65475810617763, 14.491376746189438,
 11.445523142259598,    10.295630140987,  11.74734012447073,
 11.489125293076057,   8.54400374531753,  9.591663046625438]
    return data_frame


def parse_args():
    parser = argparse.ArgumentParser(description="PV251 project - GLS regression")
    parser.add_argument('--path', nargs=1)
    parser.add_argument('--cols', default=['x', 'y'], nargs='+',
                        help='column names, expected two column names for x and Y')

    parsed = parser.parse_args(sys.argv[1:])
    return parsed.path, parsed.cols


if __name__ == '__main__':
    #
    # args = parse_args()
    # path = args[0]
    # x_name, y_name = args[1]
    path = None
    x_name = 'Year, 1956 = 0'
    y_name = 'Birds (sqrt)'

    data_frame = None
    if path == None:
        data_frame = hardwired_data()
    else:
        data_frame = pd.read_csv(path)
        data_frame = data_frame[[x_name, y_name]].copy()

    layout = Layout(data_frame, x_name, y_name)

    curdoc().add_root(layout.layout)
