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
import sqlite3

# Using tkinter filedialog to select multiple LAS files for the combination process
root = Tk()
filename = filedialog.askopenfilename(title="Choose well Log ASCII Standard (LAS) files to be combined",
                                          filetype=(("LAS Files","*.LAS *.las"),("All Files","*.*")),
                                          multiple=True)
root.destroy()

dfc = pd.DataFrame()
num = 1
for i in filename:
    try:
        las = lasio.read(i)
        df = las.df()        
        nullValue = las.well.NULL['value']
        df = df.where(df != nullValue,np.nan)
        df.insert(0,'WELLNAME','WELL-'+str(num)) #las.well.WELL['value'])
        dfc = dfc.append(df) 
        print('Reading {}'.format(i))
        num = num + 1
    except:
        print('Problem opening the file {}'.format(i))    
dfc.reset_index(inplace=True)


db = sqlite3.connect('combinedLAS.db')
dfc.to_sql('all_LAS', db, if_exists='replace')

db.close()



