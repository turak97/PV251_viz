
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from bokeh.models import PointDrawTool
from bokeh.layouts import row, column

import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
from scipy.linalg import toeplitz
from pandas import DataFrame
import numpy as np
import pandas as pd
from scipy.stats import norm


LINE_DETAIL = 5
PLOT_SIZE = 600
EXTRA_PLOT_SIZE = int(PLOT_SIZE / 2)
DATASET_COLOR = 'gold'
OLS_COLOR = 'mediumblue'
GLS_COLOR = 'limegreen'

# TODO: opravit qq rezidua u GLS
# TODO: refactoring inicializace malych plotu (spolecnou cast vytvoreni plotu vyclenit)

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

        self._OLS = None
        self._GLS = None
        self._refresh_models()  # initialize self._OLS and self._GLS
        self._main_figure = self._init_main_figure(self._data_source, x_name, y_name)

        self._OLS_qq = self._init_qq(self._OLS.resid, OLS_COLOR)
        self._GLS_qq = self._init_qq(self._GLS.resid, GLS_COLOR)

        self._OLS_index = self._init_index_plot(self._OLS.resid, OLS_COLOR)
        self._GLS_index = self._init_index_plot(self._GLS.resid, GLS_COLOR)

        self._OLS_res_res = self._init_res_res(self._OLS.resid, OLS_COLOR)
        self._GLS_res_res = self._init_res_res(self._GLS.resid, GLS_COLOR)

        ols_extra_graphs = column(
            row(self._OLS_qq, self._OLS_index, self._OLS_res_res),
            row(self._GLS_qq, self._GLS_index, self._GLS_res_res)
        )

        self.layout = row(self._main_figure, ols_extra_graphs)

    def _init_main_figure(self, data_source, x_name, y_name):
        """Initialize figure with main data and regression lines"""
        main_figure = figure(match_aspect=True, tools="pan,wheel_zoom,save,reset,box_zoom,lasso_select",
                             plot_width=PLOT_SIZE, plot_height=PLOT_SIZE, x_axis_label=x_name, y_axis_label=y_name)

        move_circle = main_figure.circle(source=data_source, x='x', y='y',
                                         size=8, fill_color=DATASET_COLOR, line_color='black')

        point_draw_tool = PointDrawTool(renderers=[move_circle], empty_value='black', add=True)
        main_figure.add_tools(point_draw_tool)

        x_ols, y_ols = self._predict(
            self._OLS,
            min(self._data_source.data['x']),
            max(self._data_source.data['x']))
        main_figure.line(x_ols, y_ols, line_color=OLS_COLOR)  # OLS line is in renderer 1
        x_gls, y_gls = self._predict(
            self._GLS,
            min(self._data_source.data['x']),
            max(self._data_source.data['x']))
        main_figure.line(x_gls, y_gls, line_color=GLS_COLOR)  # GLS line is in renderer 2

        return main_figure

    def _init_res_res(self, resid, color):
        res_res_plot = figure(plot_width=EXTRA_PLOT_SIZE, plot_height=EXTRA_PLOT_SIZE,
                            x_axis_label="Rezidua r_1, ..., r_n-1", y_axis_label="Rezidua r_2, ..., r_n")
        res_res_plot.yaxis.minor_tick_line_color = None
        res_res_plot.xaxis.minor_tick_line_color = None
        res_res_plot.toolbar_location = None

        r1 = resid[:-1].tolist()
        r1_c = sm.add_constant(r1)
        r2 = resid[1:].tolist()

        xx, yy = self._predict(
            sm.OLS(r2, r1_c).fit(),
            min(r1),
            max(r1),
        )
        res_res_plot.circle(x=r1, y=r2, size=7, color=color)
        res_res_plot.line(x=xx, y=yy, color=color)
        return res_res_plot

    def _init_index_plot(self, resid, color):
        index_plot = figure(plot_width=EXTRA_PLOT_SIZE, plot_height=EXTRA_PLOT_SIZE,
                            x_axis_label="Indexy", y_axis_label="Rezidua")
        index_plot.yaxis.minor_tick_line_color = None
        index_plot.xaxis.minor_tick_line_color = None
        index_plot.toolbar_location = None
        index_list = [x for x in range(len(resid))]
        index_plot.circle(x=index_list, y=resid, size=7, color=color)
        index_plot.line(x=index_list, y=resid, color=color)
        return index_plot

    def _init_qq(self, resid, color):
        qq_plot = figure(plot_width=EXTRA_PLOT_SIZE, plot_height=EXTRA_PLOT_SIZE,
                         x_axis_label="Theoretical Quantiles", y_axis_label="Sample Quantiles")
        qq_plot.yaxis.minor_tick_line_color = None
        qq_plot.xaxis.minor_tick_line_color = None
        qq_plot.toolbar_location = None
        qq_x, qq_y = self._get_qq_data(resid)
        qq_plot.circle(x=qq_x, y=qq_y, color=color)  # renderer 0

        x_from, x_to, y_from, y_to = self._get_qq_line(qq_x, qq_y)
        qq_plot.line(x=[x_from, x_to], y=[y_from, y_to], color='black')  # renderer 1
        return qq_plot

    def _get_qq_line(self, qq_x, qq_y):
        # qq_line gained as a line get with 0.25th and 0.75th residual points
        # from two points are calculated line parameters y = p*x + q
        x1, x2 = qq_x[int(len(qq_x) * 0.25)], qq_x[int(len(qq_x) * 0.75)]
        y1, y2 = qq_y[int(len(qq_y) * 0.25)], qq_y[int(len(qq_y) * 0.75)]
        p = (y2 - y1)/(x2 - x1)
        q = y2 - p * x2

        x_from = qq_x[0]
        x_to = qq_x[len(qq_x) - 1]
        y_from = p * x_from + q,
        y_to = p * x_to + q
        return x_from, x_to, y_from, y_to

    def _get_qq_data(self, residuals):
        df_res = pd.DataFrame(sorted(residuals), columns=['residuals'])
        res_mean = df_res['residuals'].mean()
        res_std = df_res['residuals'].std()
        df_res['z_actual'] = (df_res['residuals']).map(
            lambda x: (x - res_mean/res_std)
        )
        df_res['rank'] = df_res.index + 1
        df_res['percentile'] = df_res['rank'].map(
            lambda x: x/(len(df_res['residuals']) + 1)
        )
        df_res['z_theoretical'] = norm.ppf(df_res['percentile'])
        return df_res['z_theoretical'], df_res['z_actual']

    def _refresh_trigger(self, attr, old, new):
        self._place_last_added_correctly()
        self._refresh_models()
        self._refresh_figures()
        self._refresh_extra_graphs()

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

    def _refresh_qq_plot(self, plot, resid):
        qq_new_x, qq_new_y = self._get_qq_data(resid)
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

    def _refresh_index_plot(self, plot, resid):
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

    def _refresh_res_res_plot(self, plot, resid):
        r1 = resid[:-1].tolist()
        r2 = resid[1:].tolist()
        plot.renderers[0].data_source.update(
            data=dict(
                x=r1,
                y=r2
            )
        )

        r1_c = sm.add_constant(r1)
        xx, yy = self._predict(
            sm.OLS(r2, r1_c).fit(),
            min(r1),
            max(r1),
        )
        plot.renderers[1].data_source.update(
            data=dict(
                x=xx,
                y=yy
            )
        )

    def _refresh_extra_graphs(self):
        self._refresh_qq_plot(self._OLS_qq, self._OLS.resid)
        self._refresh_qq_plot(self._GLS_qq, self._GLS.resid)

        self._refresh_index_plot(self._OLS_index, self._OLS.resid)
        self._refresh_index_plot(self._GLS_index, self._GLS.resid)

        self._refresh_res_res_plot(self._OLS_res_res, self._OLS.resid)
        self._refresh_res_res_plot(self._GLS_res_res, self._GLS.resid)

    def _refresh_models(self):
        X, Y = self._prepare_data_for_fit()
        X = sm.add_constant(X)

        self._OLS = self._refresh_ols(X, Y)
        self._GLS = self._refresh_gls(X, Y)

    def _prepare_data_for_fit(self):
        df = self._data_source.to_df()
        X = df['x'].tolist()
        Y = df['y'].tolist()

        return X, Y

    def _refresh_ols(self, X, Y):
        return sm.OLS(Y, X).fit()

    def _refresh_gls(self, X, Y):
        ols_resid = sm.OLS(Y, X).fit().resid  # rezidua OLS
        res_fit = sm.OLS(ols_resid[1:], ols_resid[:-1]).fit()  # vypocet korelace mezi rezidui
        rho = res_fit.params  # autoregresni parametr

        order = toeplitz(np.arange(len(ols_resid)))  # some magic
        sigma = rho ** order

        return sm.GLS(Y, X, sigma=sigma).fit()

    def _refresh_figures(self):
        x_ols_new, y_ols_new = self._predict(
            self._OLS,
            min(self._data_source.data['x']),
            max(self._data_source.data['x']))
        self._main_figure.renderers[1].data_source.update(
            data=dict(
                x=x_ols_new,
                y=y_ols_new
            )
        )

        x_gls_new, y_gls_new = self._predict(
            self._GLS,
            min(self._data_source.data['x']),
            max(self._data_source.data['x']))
        self._main_figure.renderers[2].data_source.update(
            data=dict(
                x=x_gls_new,
                y=y_gls_new
            )
        )

    def _predict(self, fitted_model, min_x, max_x):
        pred_x_raw = np.linspace(
            min_x,
            max_x,
            LINE_DETAIL)
        pred_x = sm.add_constant(pred_x_raw)
        pred_y = fitted_model.predict(pred_x)
        return pred_x_raw, pred_y
