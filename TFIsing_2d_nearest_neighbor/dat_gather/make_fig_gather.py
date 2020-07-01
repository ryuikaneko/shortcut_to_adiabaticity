#!/usr/bin/env python

# coding:utf-8
from __future__ import print_function
#import sys
import re
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

all_files = glob.glob('../dat_N*/dat')
#all_files = glob.glob('../dat_N*/dat*')

list_N = []
list_time = []
list_MF_valSx = []
list_MF_valSy = []
list_MF_valSz = []
list_ED_valSx = []
list_ED_valSy = []
list_ED_valSz = []
for file_name in all_files:
#    print(file_name)
#    N = int(re.sub(".*dat_N","",file_name))
    N = int(re.sub("/dat","",re.sub(".*dat_N","",file_name)))
    list_N.append(N)
    file = open(file_name)
    lines = file.readlines()
    file.close()
    for line in lines:
        if line.startswith("list_time ["):
            line_time = (line[:-1].replace("list_time [","")).replace("]","")
            list_time.append(np.fromstring(line_time,dtype=np.float,sep=','))
        if line.startswith("list_MF_valSx ["):
            line_MF_valSx = (line[:-1].replace("list_MF_valSx [","")).replace("]","")
            list_MF_valSx.append(np.fromstring(line_MF_valSx,dtype=np.float,sep=','))
        if line.startswith("list_MF_valSy ["):
            line_MF_valSy = (line[:-1].replace("list_MF_valSy [","")).replace("]","")
            list_MF_valSy.append(np.fromstring(line_MF_valSy,dtype=np.float,sep=','))
        if line.startswith("list_MF_valSz ["):
            line_MF_valSz = (line[:-1].replace("list_MF_valSz [","")).replace("]","")
            list_MF_valSz.append(np.fromstring(line_MF_valSz,dtype=np.float,sep=','))
        if line.startswith("list_ED_valSx ["):
            line_ED_valSx = (line[:-1].replace("list_ED_valSx [","")).replace("]","")
            list_ED_valSx.append(np.fromstring(line_ED_valSx,dtype=np.float,sep=','))
        if line.startswith("list_ED_valSy ["):
            line_ED_valSy = (line[:-1].replace("list_ED_valSy [","")).replace("]","")
            list_ED_valSy.append(np.fromstring(line_ED_valSy,dtype=np.float,sep=','))
        if line.startswith("list_ED_valSz ["):
            line_ED_valSz = (line[:-1].replace("list_ED_valSz [","")).replace("]","")
            list_ED_valSz.append(np.fromstring(line_ED_valSz,dtype=np.float,sep=','))


#print(list_time)
#print(list_ED_valSz)

numN=len(list_N)
#numN=len(list_N)-1

f = open("dat_MF_valSx","w")
for j in range(len(list_time[0])):
    f.write("{} ".format(list_time[0][j]))
    for i in range(numN):
        f.write("{} ".format(list_MF_valSx[i][j]))
    f.write("\n")
f.close()

f = open("dat_MF_valSy","w")
for j in range(len(list_time[0])):
    f.write("{} ".format(list_time[0][j]))
    for i in range(numN):
        f.write("{} ".format(list_MF_valSy[i][j]))
    f.write("\n")
f.close()

f = open("dat_MF_valSz","w")
for j in range(len(list_time[0])):
    f.write("{} ".format(list_time[0][j]))
    for i in range(numN):
        f.write("{} ".format(list_MF_valSz[i][j]))
    f.write("\n")
f.close()

f = open("dat_ED_valSx","w")
for j in range(len(list_time[0])):
    f.write("{} ".format(list_time[0][j]))
    for i in range(numN):
        f.write("{} ".format(list_ED_valSx[i][j]))
    f.write("\n")
f.close()

f = open("dat_ED_valSy","w")
for j in range(len(list_time[0])):
    f.write("{} ".format(list_time[0][j]))
    for i in range(numN):
        f.write("{} ".format(list_ED_valSy[i][j]))
    f.write("\n")
f.close()

f = open("dat_ED_valSz","w")
for j in range(len(list_time[0])):
#    print(list_time[0][j],end=" ")
    f.write("{} ".format(list_time[0][j]))
    for i in range(numN):
#        print(list_ED_valSz[i][j],end=" ")
        f.write("{} ".format(list_ED_valSz[i][j]))
#    print("")
    f.write("\n")
f.close()


fig10 = plt.figure()
fig10.suptitle("<S^x>/S")
for i in range(numN):
    plt.plot(list_time[i],list_ED_valSx[i],label="N"+str(list_N[i]))
plt.xlabel("time")
#plt.legend(bbox_to_anchor=(1,0),loc='lower right',borderaxespad=1)
#plt.legend(bbox_to_anchor=(1,1),loc='upper right',borderaxespad=1)
plt.legend(bbox_to_anchor=(0,1),loc='upper left',borderaxespad=1)
#plt.gca().invert_xaxis()
fig10.savefig("fig_ED_valSx.png")

fig20 = plt.figure()
fig20.suptitle("<S^y>/S")
for i in range(numN):
    plt.plot(list_time[i],list_ED_valSy[i],label="N"+str(list_N[i]))
plt.xlabel("time")
#plt.legend(bbox_to_anchor=(1,0),loc='lower right',borderaxespad=1)
#plt.legend(bbox_to_anchor=(1,1),loc='upper right',borderaxespad=1)
plt.legend(bbox_to_anchor=(0,1),loc='upper left',borderaxespad=1)
#plt.gca().invert_xaxis()
fig20.savefig("fig_ED_valSy.png")

fig30 = plt.figure()
fig30.suptitle("<S^z>/S")
for i in range(numN):
    plt.plot(list_time[i],list_ED_valSz[i],label="N"+str(list_N[i]))
plt.xlabel("time")
#plt.legend(bbox_to_anchor=(1,0),loc='lower right',borderaxespad=1)
#plt.legend(bbox_to_anchor=(1,1),loc='upper right',borderaxespad=1)
plt.legend(bbox_to_anchor=(0,1),loc='upper left',borderaxespad=1)
#plt.gca().invert_xaxis()
fig30.savefig("fig_ED_valSz.png")
