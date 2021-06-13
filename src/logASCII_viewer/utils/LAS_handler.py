"""
Created on Thu Dec 24 15:35:42 2020

@author: imran.fadhil
"""

import lasio
import pandas as pd
import numpy as np
from tkinter import filedialog
from tkinter import *
import streamlit as st
from io import StringIO


def read_las_file(las_file, unit='metric'):
    merged_data = pd.DataFrame()
    header_data = pd.DataFrame()
    for f in las_file:
        bytes_data = f.read()
        str_io = StringIO(bytes_data.decode('Windows-1252'))
        las = lasio.read(str_io)
        df = las.df()

        if unit == 'imperial':
            df['DEPTH'] = las.depth_ft
        elif unit == 'metric':
            df['DEPTH'] = las.depth_m

        nullValue = las.well.NULL['value']
        df = df.where(df != nullValue, np.nan)

        wellname = las.well.WELL['value']
        df.insert(0, 'WELLNAME', wellname)   

        fieldname = las.well.FLD['value']
        if fieldname is not '':
            df.insert(0, 'FIELDNAME', fieldname)
        else:
            fieldname = wellname.split('-')[0]
            df.insert(0, 'FIELDNAME', fieldname)
        
        merged_data = merged_data.append(df)

        well_dict = [{x:las.well[x]['value'] for x in las.well.keys()}]
        well_header = pd.DataFrame(well_dict)        
        
        header_data = header_data.append(well_header)
    
    header_data_cols = header_data.columns
    header_data = header_data.transpose()
    header_data.rename({0:wellname}, axis=1, inplace=True)
    header_data.rename({x:v for x,v in enumerate(header_data_cols)}, axis=0, inplace=True)

    merged_data.reset_index(inplace=True, drop=True)
    return merged_data, header_data


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

    dfc.to_parquet('../data/combined_LAS.pqt')
