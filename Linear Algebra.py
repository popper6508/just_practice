import os
import glob
import unicodedata
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as pltp
import numpy as np
import requests
from sklearn.linear_model import LinearRegression
import json
import random
import xml.etree.ElementTree as et
from bs4 import BeautifulSoup

def vector_add(a,b) :
  a_dim = len(a.columns)
  b_dim = len(b.columns)
  if a_dim != b_dim :
    print("Size mismatch")
  else : 
    c = a + b
    print(c)

x1 = pd.DataFrame([[1,4,2,3], [6,1,0,4]])
x2 = pd.DataFrame([[7,9,4,6], [4,6,3,9]])

vector_add(x1,x2)

def vector_prob(a,b) :
  a_dim = len(a.columns)
  b_dim = len(b.index)
  if a_dim != b_dim :
    print("Size mismatch")
  else :
    c = np.matmul(a,b)
    print(c)

def lu_decomp(mat) :
   rows, cols = np.shape(mat)
   if rows < cols :
     s = rows
     l = cols
   else :
     s = cols
     l = rows
   for k in range(0,s) :
     x = 1.0 / mat[k,k]
     for i in range(k+1,rows) :
       mat[i,k] = mat[i,k] * x
     for i in range(k+1,rows) :
        for j in range(k+1,cols) :
          mat[i,j] = mat[i,j] - mat[i,k] * mat[k,j]
   return(mat)

def fst_equ(ma, y) :
  row, col = np.shape(ma)
  mat_1 = lu_decomp(ma)
  z = []
  x = []
  if (row != col)|(row != len(y)) :
    print("Inapproporiate type marrix")
  else : 
    mat_2 = np.zeros((row, col))
    mat_3 = np.zeros((row, col))

    for i in range(row) :
      mat_2[i,i] = 1
      for j in range(i,col) :
        mat_3[i,j] = mat_1[i,j]

    for i in range(1,row) :
      for j in range(i) :
        mat_2[i,j] = mat_1[i,j]
        
    #z value
    z.append(y[0])
    for i in range(1,row) :
      a = y[i]
      for j in range(i) :
        a -= z[i-1]*mat_2[i,j]
      z.append(a)

    x.append(z[0]/mat_3[row-1,col-1])

    for i in range(1,row) :
      b = z[row-1-i]
      for j in range(i) :
        b += -x[j]*mat_3[row-1-i,col-j-1]
        b = b/mat_3[row-i-1,col-i-1]
      x.append(b)
    return(x)

