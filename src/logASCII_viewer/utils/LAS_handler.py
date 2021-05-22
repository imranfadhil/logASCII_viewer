# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 15:35:42 2020

@author: imran.fadhil
"""

import lasio
import pandas as pd
import numpy as np
from tkinter import filedialog
from tkinter import *

from io import StringIO


def read_las_file(las_file):
    merged_data = pd.DataFrame()
    for f in las_file:
        bytes_data = f.read()
        str_io = StringIO(bytes_data.decode('Windows-1252'))
        las = lasio.read(str_io)
        df = las.df()
        nullValue = las.well.NULL['value']
        df = df.where(df != nullValue, np.nan)
        df.insert(0, 'WELLNAME', las.well.WELL['value'])
        df.index = np.around(df.index.tolist(), 1)
        df['DEPTH'] = df.index
        merged_data = merged_data.append(df)
    merged_data.reset_index(inplace=True, drop=True)
    return merged_data


def get_header_info():
    return


if __name__ == '__main__':
    # Using tkinter filedialog to select multiple LAS files for the combination process
    root = Tk()
    filename = filedialog.askopenfilename(title="Choose well Log ASCII Standard (LAS) files to be combined",
                                          filetype=(
                                              ("LAS Files", "*.LAS *.las"), ("All Files", "*.*")),
                                          multiple=True)
    root.destroy()

    dfc = pd.DataFrame()
    num = 1
    for i in filename:
        try:
            las = lasio.read(i)
            df = las.df()
            nullValue = las.well.NULL['value']
            df = df.where(df != nullValue, np.nan)
            # 'WELL-'+str(num)) #
            df.insert(0, 'WELLNAME', las.well.WELL['value'])
            dfc = dfc.append(df)
            print('Reading {}'.format(i))
            num = num + 1
        except:
            print('Problem opening the file {}'.format(i))
    dfc.reset_index(inplace=True)
    print('\n ...Finish reading %d LAS files...\n' % num)

    dfc.to_parquet('../data/LOGIC_LAS.pqt')
