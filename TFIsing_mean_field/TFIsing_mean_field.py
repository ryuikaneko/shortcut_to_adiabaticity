#!/usr/bin/env python

# coding:utf-8
from __future__ import print_function
import math
import numpy as np
#import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
#import scipy as scipy
#import scipy.integrate as integrate
import argparse
import time
#import copy
#import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='shortcut to adiabaticity TFI mean field')
    parser.add_argument('-twoS',metavar='twoS',dest='twoS',type=np.int,default=1000,help='set 2S')
    parser.add_argument('-tau',metavar='tau',dest='tau',type=np.float64,default=1.0,help='set tau (total time)')
    return parser.parse_args()

def make_spin(twoS):
    Ssize = 0.5*twoS
    Matsize = twoS+1
    npS0 = np.identity(Matsize)
    npSz = np.diag([Ssize-i for i in range(Matsize)])
    npSp = np.diag([np.sqrt(Ssize*(Ssize+1)-((Ssize-i)-1)*(Ssize-i)) for i in range(Matsize-1)],k=1)
    npSm = np.diag([np.sqrt(Ssize*(Ssize+1)-((Ssize-i)-1)*(Ssize-i)) for i in range(Matsize-1)],k=-1)
    npSx = 0.5*(npSp+npSm)
    npSy = -0.5*1j*(npSp-npSm)
#    print("npS0",npS0)
#    print("npSz",npSz)
#    print("npSp",npSp)
#    print("npSm",npSm)
#    print("npSx",npSx)
#    print("npSy",npSy)
    S0 = scipy.sparse.csr_matrix(npS0)
    Sz = scipy.sparse.csr_matrix(npSz)
    Sx = scipy.sparse.csr_matrix(npSx)
    Sy = scipy.sparse.csr_matrix(npSy)
#    print("S0",S0)
#    print("Sx",Sx)
#    print("Sy",Sy)
#    print("Sz",Sz)
    return S0,Sx,Sy,Sz

def calc_Gamma(t,tau):
    tt = t/tau
    return 48.0*tt**5 - 120.0*tt**4 + 100.0*tt**3 - 30.0*tt**2 + 2.0

def calc_GammaDot(t,tau):
    tt = t/tau
    return (48.0*5*tt**4 - 120.0*4*tt**3 + 100.0*3*tt**2 - 30.0*2*tt)/tau

def calc_Mz(J,Hz,Gamma):
    Mz = 0.5
    for i in range(10000):
        newMz = (J*Mz+Hz)/np.sqrt((J*Mz+Hz)**2+Gamma**2)
        if np.abs(newMz-Mz)<1e-12:
            break
        Mz = newMz
    if i>=10000-1:
        newMz = (J*Mz+Hz)/np.sqrt((J*Mz+Hz)**2+Gamma**2)
        print("### !!! Mz error may be large",np.abs(newMz-Mz))
    return Mz

def calc_MzDot(J,Hz,Gamma,GammaDot,Mz):
    numer = - Gamma*GammaDot*Mz**2
    denom = 2.0*J**2*Mz**3 + 3.0*J*Hz*Mz**2 - (J**2-Hz**2-Gamma**2)*Mz - J*Hz
    return numer/denom

def calc_ThDot(J,Hz,Gamma,GammaDot,Mz,MzDot):
    numer = (J*Mz+Hz)*GammaDot - J*MzDot*Gamma
    denom = (J*Mz+Hz)**2 + Gamma**2
    return 0.5*numer/denom

def make_interaction(tau,J,Hz,list_time):
    list_Gamma = [calc_Gamma(t,tau) for t in list_time]
    list_GammaDot = [calc_GammaDot(t,tau) for t in list_time]
    list_Mz = [calc_Mz(J,Hz,list_Gamma[i]) for i,t in enumerate(list_time)]
    list_MzDot = [calc_MzDot(J,Hz,list_Gamma[i],list_GammaDot[i],list_Mz[i]) for i,t in enumerate(list_time)]
    list_ThDot = [calc_ThDot(J,Hz,list_Gamma[i],list_GammaDot[i],list_Mz[i],list_MzDot[i]) for i,t in enumerate(list_time)]
    return list_Gamma, list_GammaDot, list_Mz, list_MzDot, list_ThDot

def make_ham(S0,Sx,Sy,Sz,twoS,J,Gamma,Hz,ThDot):
    Ham = - J/(0.5*twoS)*Sz.dot(Sz) - 2.0*Gamma*Sx - 2.0*Hz*Sz + 2.0*ThDot*Sy
#    print(Ham)
#    print(Ham.todense())
    return Ham

def calc_physquant(S0,Sx,Sy,Sz,invS,Ham,psi):
    norm2 = np.linalg.norm(psi)**2
    valSx = (psi.conj().T).dot(Sx.dot(psi)).real*invS/norm2
    valSy = (psi.conj().T).dot(Sy.dot(psi)).real*invS/norm2
    valSz = (psi.conj().T).dot(Sz.dot(psi)).real*invS/norm2
    valHam = (psi.conj().T).dot(Ham.dot(psi)).real/norm2
#    print("norm^2",norm2)
#    print("valSx",valSx)
#    print("valSy",valSy)
#    print("valSz",valSz)
#    print("valHam",valHam)
    return norm2, valSx, valSy, valSz, valHam

#----

def main():
    np.set_printoptions(threshold=10000)
    args = parse_args()
    twoS = args.twoS
    tau = args.tau
#
    invS = 2.0/twoS
    dt = tau/1000.0
    time_i = 0.0
    time_f = tau
    Nsteps = int(tau/dt+0.1)+1
    list_time = [time_i+i*(time_f-time_i)/(Nsteps-1) for i in range(Nsteps)]
#    list_time_for_U = [time_i+(i+0.5)*(time_f-time_i)/(Nsteps-1) for i in range(Nsteps)]
#
    J = 1.0
    Hz = 1e-3
#
    print("twoS=",twoS)
    print("tau=",tau)
    print("dt=",dt)
    print("Nsteps=",Nsteps)
    print("time_i=",time_i)
    print("time_f=",time_f)
    print("list_time: t=",list_time)
#    print("list_time_for_U: t+dt/2=",list_time_for_U)
    print("J",J)
    print("Hz",Hz)

## prepare spin and interaction
    start = time.time()
    S0, Sx, Sy, Sz = make_spin(twoS)
    list_Gamma, list_GammaDot, list_Mz, list_MzDot, list_ThDot \
        = make_interaction(tau,J,Hz,list_time)
    print("list_Gamma",list_Gamma)
    print("list_GammaDot",list_GammaDot)
    print("list_Mz",list_Mz)
    print("list_MzDot",list_MzDot)
    print("list_ThDot",list_ThDot)
    end = time.time()
    print("time: prepare spin and interaction",end - start)

## prepare initial state
    start = time.time()
    J = 1.0; Gamma = 2.0; Hz = 1e-3; ThDot = 0.0
    Ham = make_ham(S0,Sx,Sy,Sz,twoS,J,Gamma,Hz,ThDot)
    if twoS<10:
        ene,vec = scipy.linalg.eigh(Ham.todense())
    else:
        ene,vec = scipy.sparse.linalg.eigsh(Ham,which='SA',k=2)
    print("energy(t=0)",ene[0],ene[1])
#    print("vector(t=0)",vec[:,0],vec[:,1])
    list_norm2 = []
    list_valSx = []
    list_valSy = []
    list_valSz = []
    list_valHam = []
    i = 0
    t = list_time[i]
#    print("i",i)
#    print("t",t)
    norm2, valSx, valSy, valSz, valHam = calc_physquant(S0,Sx,Sy,Sz,invS,Ham,vec[:,0])
    list_norm2.append(norm2)
    list_valSx.append(valSx)
    list_valSy.append(valSy)
    list_valSz.append(valSz)
    list_valHam.append(valHam)
    end = time.time()
    print("time: prepare intial state",end - start)

## calculate dynamics
    start = time.time()
    psi = vec[:,0]
    for i in range(1,Nsteps):
        t = list_time[i]
        Gamma = list_Gamma[i]
#        GammaDot = list_GammaDot[i]
#        Mz = list_Mz[i]
#        MzDot = list_MzDot[i]
        ThDot = list_ThDot[i]
        Ham = make_ham(S0,Sx,Sy,Sz,twoS,J,Gamma,Hz,ThDot)
#        HamExact = make_ham(S0,Sx,Sy,Sz,twoS,J,Gamma,Hz,0.0)
        psi = (scipy.sparse.linalg.expm_multiply((-1j)*dt*Ham,psi,start=0.0,stop=1.0,num=2,endpoint=True))[1]
#        print("i",i)
#        print("t",t)
#        print("psi",psi)
        norm2, valSx, valSy, valSz, valHam = calc_physquant(S0,Sx,Sy,Sz,invS,Ham,psi)
        list_norm2.append(norm2)
        list_valSx.append(valSx)
        list_valSy.append(valSy)
        list_valSz.append(valSz)
        list_valHam.append(valHam)
    end = time.time()
    print("time: calculate dynamics",end - start)

## plot evolution
    start = time.time()
#
    fig10 = plt.figure()
    fig10.suptitle("Gamma")
    plt.plot(list_time,list_Gamma)
    plt.xlabel("time (from 0 to tau)")
    fig10.savefig("fig_Gamma.png")
#
    fig20 = plt.figure()
    fig20.suptitle("GammaDot")
    plt.plot(list_time,list_GammaDot)
    plt.xlabel("time (from 0 to tau)")
    fig20.savefig("fig_GammaDot.png")
#
    fig30 = plt.figure()
    fig30.suptitle("Mz")
    plt.plot(list_time,list_Mz)
    plt.xlabel("time (from 0 to tau)")
    fig30.savefig("fig_Mz.png")
#
    fig40 = plt.figure()
    fig40.suptitle("MzDot")
    plt.plot(list_time,list_MzDot)
    plt.xlabel("time (from 0 to tau)")
    fig40.savefig("fig_MzDot.png")
#
    fig50 = plt.figure()
    fig50.suptitle("ThDot")
    plt.plot(list_time,list_ThDot)
    plt.xlabel("time (from 0 to tau)")
    fig50.savefig("fig_ThDot.png")
#
    fig60 = plt.figure()
    fig60.suptitle("norm^2")
    plt.plot(list_time,list_norm2)
    plt.xlabel("time (from 0 to tau)")
    fig60.savefig("fig_norm2.png")
#
    fig70 = plt.figure()
    fig70.suptitle("<S^x>/S")
    plt.plot(list_time,list_valSx)
    plt.xlabel("time (from 0 to tau)")
    fig70.savefig("fig_valSx.png")
#
    fig80 = plt.figure()
    fig80.suptitle("<S^y>/S")
    plt.plot(list_time,list_valSy)
    plt.xlabel("time (from 0 to tau)")
    fig80.savefig("fig_valSy.png")
#
    fig90 = plt.figure()
    fig90.suptitle("<S^z>/S")
    plt.plot(list_time,list_valSz)
    plt.xlabel("time (from 0 to tau)")
    fig90.savefig("fig_valSz.png")
#
    fig100 = plt.figure()
    fig100.suptitle("energy")
    plt.plot(list_time,list_valHam)
    plt.xlabel("time (from 0 to tau)")
    fig100.savefig("fig_energy.png")
#
#    plt.show()
    end = time.time()
    print("time: plot",end - start)

if __name__ == "__main__":
    main()
