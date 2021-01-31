
import argparse  # to easily parse arguments
import sys  # sys.argv[1:] -- to obtain possible file path

from bokeh.server.server import Server
from bokeh.core import validation

import layout_factory as lf

import pandas as pd

import numpy as np


validation.silence(1002, True)  # silence the bokeh plot warnings


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

    args = parse_args()
    path = args[0]
    x_name, y_name = args[1]

    data_frame = None
    if path == None:
        data_frame = hardwired_data()
        x_name = 'Year, 1956 = 0'
        y_name = 'Birds (sqrt)'
    else:
        data_frame = pd.read_csv(path)
        data_frame = data_frame[[x_name, y_name]].copy()
    #
    # ##### DEL AFTER; PREPROCESSING
    # data_frame[y_name] = data_frame[y_name].apply(lambda x: np.sqrt(x))
    # data_frame[x_name] = data_frame[x_name].apply(lambda x: x - min(data_frame[x_name]))

    def bkapp(doc):

        layout = lf.Layout(data_frame, x_name, y_name)

        doc.add_root(layout.layout)

    server = Server({'/': bkapp}, num_procs=1)
    server.start()

    server.io_loop.add_callback(server.show, "/")
    try:
        server.io_loop.start()
    except KeyboardInterrupt:
        print()
        print("Session closed, have a nice day.")
