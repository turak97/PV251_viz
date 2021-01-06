
import argparse  # to easily parse arguments
import sys  # sys.argv[1:] -- to obtain possible file path

from bokeh.server.server import Server
from bokeh.core import validation

import layout_factory as lf

import pandas as pd

import numpy as np


validation.silence(1002, True)  # silence the bokeh plot warnings


def parse_args():
    parser = argparse.ArgumentParser(description="PV251 project - GLS regression")
    parser.add_argument('--path', default='datasets/Hawaii.csv', nargs=1)
    parser.add_argument('--cols', default=['x', 'y'], nargs='+', help='column names')

    parsed = parser.parse_args(sys.argv[1:])
    return parsed.path, parsed.cols


if __name__ == '__main__':

    args = parse_args()
    path = args[0]
    x_name, y_name = args[1]

    data_frame = pd.read_csv(path)
    data_frame = data_frame[[x_name, y_name]].copy()

    ##### DEL AFTER; PREPROCESSING
    data_frame[y_name] = data_frame[y_name].apply(lambda x: np.sqrt(x))
    data_frame[x_name] = data_frame[x_name].apply(lambda x: x - min(data_frame[x_name]))

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
