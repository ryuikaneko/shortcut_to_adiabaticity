#!/usr/bin/env python

# coding:utf-8
from __future__ import print_function
import math
import numpy as np
np.seterr(divide='ignore', invalid='ignore')## ignore division by 0 and nan
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
from scipy.interpolate import interp1d
from bayes_opt import BayesianOptimization

def parse_args():
    parser = argparse.ArgumentParser(description='shortcut to adiabaticity TFI: 1d with mean field schedule')
    parser.add_argument('-N',metavar='N',dest='N',type=np.int,default=8,help='set N')
#    parser.add_argument('-twoS',metavar='twoS',dest='twoS',type=np.int,default=1,help='set 2S')
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

def calc_physquant(Sx,Sy,Sz,invS,Ham,psi):
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

def calc_overlap(psi0,psi1):
    norm0 = np.linalg.norm(psi0)
    norm1 = np.linalg.norm(psi1)
    return np.abs((psi0.conj().T).dot(psi1))/norm0/norm1

#----

def make_list_1(N):
#    list_Jz = np.ones(N,dtype=np.float64)
    list_Jz = np.ones(N,dtype=np.complex128)
    return list_Jz

def make_list_2(N):
    list_site1 = [i for i in range(N)]
    list_site2 = [(i+1)%N for i in range(N)]
#    list_Jzz = np.ones(N,dtype=np.float64)
    list_Jzz = np.ones(N,dtype=np.complex128)
    return list_site1, list_site2, list_Jzz

def make_ham_1(S0,Sz,N,list_Jz):
#    Ham = scipy.sparse.csr_matrix((2**N,2**N),dtype=np.float64)
    Ham = scipy.sparse.csr_matrix((2**N,2**N),dtype=np.complex128)
    for site1 in range(N):
        ISz = 1
        Jz = list_Jz[site1]
        for site2 in range(N):
            if site2==site1:
                ISz = scipy.sparse.kron(ISz,Sz,format='csr')
            else:
                ISz = scipy.sparse.kron(ISz,S0,format='csr')
        Ham -= Jz * ISz
    return Ham

def make_ham_2(S0,Sz,N,Nbond,list_site1,list_site2,list_Jzz):
#    Ham = scipy.sparse.csr_matrix((2**N,2**N),dtype=np.float64)
    Ham = scipy.sparse.csr_matrix((2**N,2**N),dtype=np.complex128)
    for bond in range(Nbond):
        i1 = list_site1[bond]
        i2 = list_site2[bond]
        Jzz = list_Jzz[bond]
        SzSz = 1
        for site in range(N):
            if site==i1 or site==i2:
                SzSz = scipy.sparse.kron(SzSz,Sz,format='csr')
            else:
                SzSz = scipy.sparse.kron(SzSz,S0,format='csr')
        Ham -= Jzz * SzSz
    return Ham

## Ham_zz = -Sz.Sz
## Ham_x = -Sx
## Ham_y = -Sy
## Ham_z = -Sz
def make_ham_1d(Ham_zz,Ham_x,Ham_y,Ham_z,twoS,J,Gamma,Hz,ThDot):
    Ham = J/(0.5*twoS)*Ham_zz + 2.0*Gamma*Ham_x + 2.0*Hz*Ham_z - 2.0*ThDot*Ham_y
    return Ham

#----

def calc_overlap_final_state(N,twoS,tau,J,Hz,Nbond,invS,dt,Nsteps,\
    list_time,list_Gamma,list_ThDot,flag_fig_mag):
#def calc_overlap_final_state(N,twoS,tau,J,Hz,Nbond,invS,dt,Nsteps,\
#    list_time,list_Gamma,list_GammaDot,list_Mz,list_MzDot,list_ThDot,flag_fig_mag):
#    np.set_printoptions(threshold=10000)
#    args = parse_args()
#    N = args.N
##    twoS = args.twoS
#    twoS = 1 ## spin 1/2
#    tau = args.tau
#
#    Nbond = N
#    invS = 2.0/twoS
#    dt = tau/1000.0
#    time_i = 0.0
#    time_f = tau
#    Nsteps = int(tau/dt+0.1)+1
#    list_time = [time_i+i*(time_f-time_i)/(Nsteps-1) for i in range(Nsteps)]
##    list_time_for_U = [time_i+(i+0.5)*(time_f-time_i)/(Nsteps-1) for i in range(Nsteps)]
#
#    J = 1.0
#    Hz = 1e-3
#
    print("### N=",N)
    print("### Nbond=",Nbond)
    print("### twoS=",twoS)
    print("### tau=",tau)
    print("### dt=",dt)
    print("### Nsteps=",Nsteps)
#    print("time_i=",time_i)
#    print("time_f=",time_f)
##    print("list_time: t=",list_time)
#    print("list_time",list_time)
##    print("list_time_for_U: t+dt/2=",list_time_for_U)
    print("### J",J)
    print("### Hz",Hz)

## prepare spin and interaction
    start = time.time()
#---- for MF
    S0, Sx, Sy, Sz = make_spin(twoS)
#    list_Gamma, list_GammaDot, list_Mz, list_MzDot, list_ThDot \
#        = make_interaction(tau,J,Hz,list_time)
#    print("list_Gamma",list_Gamma)
#    print("list_GammaDot",list_GammaDot)
#    print("list_Mz",list_Mz)
#    print("list_MzDot",list_MzDot)
#    print("list_ThDot",list_ThDot)
#---- for ED
    list_site1, list_site2, list_Jzz = make_list_2(N)
    _, _, list_Jxx = make_list_2(N)
    list_Jx = make_list_1(N)
    list_Jy = make_list_1(N)
    list_Jz = make_list_1(N)
    print("### site1",list_site1)
    print("### site2",list_site2)
    print("### Jzz",list_Jzz)
    print("### Jx",list_Jx)
    print("### Jy",list_Jy)
    print("### Jz",list_Jz)
    Ham_zz = make_ham_2(S0,Sz,N,Nbond,list_site1,list_site2,list_Jzz) ## FM Ising: -Sz.Sz
    Ham_x = make_ham_1(S0,Sx,N,list_Jx) ## -Sx
    Ham_y = make_ham_1(S0,Sy,N,list_Jy) ## -Sy
    Ham_z = make_ham_1(S0,Sz,N,list_Jz) ## -Sz
    Op_Mx = -Ham_x/N
    Op_My = -Ham_y/N
    Op_Mz = -Ham_z/N
#----
    end = time.time()
    print("### time: prepare spin and interaction",end - start)

## prepare initial state
    start = time.time()
    Gamma = 2.0; ThDot = 0.0
#    J = 1.0; Gamma = 2.0; Hz = 1e-3; ThDot = 0.0
    Ham_MF = make_ham(S0,Sx,Sy,Sz,twoS,J,Gamma,Hz,ThDot)
    Ham_ED = make_ham_1d(Ham_zz,Ham_x,Ham_y,Ham_z,twoS,J,Gamma,Hz,ThDot)
    if twoS<10:
        ene_MF,vec_MF = scipy.linalg.eigh(Ham_MF.todense())
    else:
        ene_MF,vec_MF = scipy.sparse.linalg.eigsh(Ham_MF,which='SA',k=2)
    idx = ene_MF.argsort()
    ene_MF = ene_MF[idx]
    vec_MF = vec_MF[:,idx]
    if 2**N<10:
        ene_ED,vec_ED = scipy.linalg.eigh(Ham_ED.todense())
    else:
        ene_ED,vec_ED = scipy.sparse.linalg.eigsh(Ham_ED,which='SA',k=2)
    idx = ene_ED.argsort()
    ene_ED = ene_ED[idx]
    vec_ED = vec_ED[:,idx]
    print("### energy_MF(t=0)",ene_MF[0],ene_MF[1])
    print("### energy_ED(t=0)",ene_ED[0],ene_ED[1])
#    print("vector_MF(t=0)",vec_MF[:,0],vec_MF[:,1])
    list_MF_norm2 = []
    list_MF_valSx = []
    list_MF_valSy = []
    list_MF_valSz = []
    list_MF_valHam = []
    list_ED_norm2 = []
    list_ED_valSx = []
    list_ED_valSy = []
    list_ED_valSz = []
    list_ED_valHam = []
    i = 0
    t = list_time[i]
#    print("i",i)
#    print("t",t)
    norm2, valSx, valSy, valSz, valHam = calc_physquant(Sx,Sy,Sz,invS,Ham_MF,vec_MF[:,0])
    list_MF_norm2.append(norm2)
    list_MF_valSx.append(valSx)
    list_MF_valSy.append(valSy)
    list_MF_valSz.append(valSz)
    list_MF_valHam.append(valHam)
    norm2, valSx, valSy, valSz, valHam = calc_physquant(Op_Mx,Op_My,Op_Mz,invS,Ham_ED,vec_ED[:,0])
    list_ED_norm2.append(norm2)
    list_ED_valSx.append(valSx)
    list_ED_valSy.append(valSy)
    list_ED_valSz.append(valSz)
    list_ED_valHam.append(valHam)
    end = time.time()
    print("### time: prepare intial state",end - start)

## calculate dynamics
    start = time.time()
    psi_MF = vec_MF[:,0]
    psi_ED = vec_ED[:,0]
    if flag_fig_mag == 1:
        for i in range(1,Nsteps):
            t = list_time[i]
            Gamma = list_Gamma[i]
#            GammaDot = list_GammaDot[i]
#            Mz = list_Mz[i]
#            MzDot = list_MzDot[i]
            ThDot = list_ThDot[i]
            Ham_MF = make_ham(S0,Sx,Sy,Sz,twoS,J,Gamma,Hz,ThDot)
            Ham_ED = make_ham_1d(Ham_zz,Ham_x,Ham_y,Ham_z,twoS,J,Gamma,Hz,ThDot)
            psi_MF = (scipy.sparse.linalg.expm_multiply((-1j)*dt*Ham_MF,psi_MF,start=0.0,stop=1.0,num=2,endpoint=True))[1]
            psi_ED = (scipy.sparse.linalg.expm_multiply((-1j)*dt*Ham_ED,psi_ED,start=0.0,stop=1.0,num=2,endpoint=True))[1]
#            print("i",i)
#            print("t",t)
#            print("psi_MF",psi_MF)
#            print("psi_ED",psi_ED)
            norm2, valSx, valSy, valSz, valHam = calc_physquant(Sx,Sy,Sz,invS,Ham_MF,psi_MF)
            list_MF_norm2.append(norm2)
            list_MF_valSx.append(valSx)
            list_MF_valSy.append(valSy)
            list_MF_valSz.append(valSz)
            list_MF_valHam.append(valHam)
            norm2, valSx, valSy, valSz, valHam = calc_physquant(Op_Mx,Op_My,Op_Mz,invS,Ham_ED,psi_ED)
            list_ED_norm2.append(norm2)
            list_ED_valSx.append(valSx)
            list_ED_valSy.append(valSy)
            list_ED_valSz.append(valSz)
            list_ED_valHam.append(valHam)
    else:
        for i in range(1,Nsteps):
            t = list_time[i]
            Gamma = list_Gamma[i]
#            GammaDot = list_GammaDot[i]
#            Mz = list_Mz[i]
#            MzDot = list_MzDot[i]
            ThDot = list_ThDot[i]
            Ham_MF = make_ham(S0,Sx,Sy,Sz,twoS,J,Gamma,Hz,ThDot)
            Ham_ED = make_ham_1d(Ham_zz,Ham_x,Ham_y,Ham_z,twoS,J,Gamma,Hz,ThDot)
            psi_MF = (scipy.sparse.linalg.expm_multiply((-1j)*dt*Ham_MF,psi_MF,start=0.0,stop=1.0,num=2,endpoint=True))[1]
            psi_ED = (scipy.sparse.linalg.expm_multiply((-1j)*dt*Ham_ED,psi_ED,start=0.0,stop=1.0,num=2,endpoint=True))[1]
#            print("i",i)
#            print("t",t)
#            print("psi_MF",psi_MF)
#            print("psi_ED",psi_ED)
#            norm2, valSx, valSy, valSz, valHam = calc_physquant(Sx,Sy,Sz,invS,Ham_MF,psi_MF)
#            list_MF_norm2.append(norm2)
#            list_MF_valSx.append(valSx)
#            list_MF_valSy.append(valSy)
#            list_MF_valSz.append(valSz)
#            list_MF_valHam.append(valHam)
#            norm2, valSx, valSy, valSz, valHam = calc_physquant(Op_Mx,Op_My,Op_Mz,invS,Ham_ED,psi_ED)
#            list_ED_norm2.append(norm2)
#            list_ED_valSx.append(valSx)
#            list_ED_valSy.append(valSy)
#            list_ED_valSz.append(valSz)
#            list_ED_valHam.append(valHam)
    end = time.time()
    print("### time: calculate dynamics",end - start)

## final state overlap
    start = time.time()
    Gamma = 0.0; ThDot = 0.0
#    J = 1.0; Gamma = 0.0; Hz = 1e-3; ThDot = 0.0
    Ham_MF = make_ham(S0,Sx,Sy,Sz,twoS,J,Gamma,Hz,ThDot)
    Ham_ED = make_ham_1d(Ham_zz,Ham_x,Ham_y,Ham_z,twoS,J,Gamma,Hz,ThDot)
#
    norm2, valSx, valSy, valSz, valHam = calc_physquant(Sx,Sy,Sz,invS,Ham_MF,psi_MF)
    print("### RTE energy_MF(t=tau)",valHam)
    norm2, valSx, valSy, valSz, valHam = calc_physquant(Op_Mx,Op_My,Op_Mz,invS,Ham_ED,psi_ED)
    print("### RTE energy_ED(t=tau)",valHam)
#
    print("### Ham_MF hermitian?",np.allclose(Ham_MF.todense(),Ham_MF.todense().T.conj(),rtol=1e-6,atol=1e-6))
    print("### Ham_ED hermitian?",np.allclose(Ham_ED.todense(),Ham_ED.todense().T.conj(),rtol=1e-6,atol=1e-6))
    if twoS<10:
        print("### Ham_MF scipy.linalg.eigh")
        ene_MF,vec_MF = scipy.linalg.eigh(Ham_MF.todense())
    else:
        print("### Ham_MF scipy.sparse.linalg.eigsh")
        ene_MF,vec_MF = scipy.sparse.linalg.eigsh(Ham_MF,which='SA',k=2)
    idx = ene_MF.argsort()
    ene_MF = ene_MF[idx]
    vec_MF = vec_MF[:,idx]
    if 2**N<10:
        print("### Ham_ED scipy.linalg.eigh")
        ene_ED,vec_ED = scipy.linalg.eigh(Ham_ED.todense())
    else:
        print("### Ham_ED scipy.sparse.linalg.eigsh")
        ene_ED,vec_ED = scipy.sparse.linalg.eigsh(Ham_ED,which='SA',k=2)
    idx = ene_ED.argsort()
    ene_ED = ene_ED[idx]
    vec_ED = vec_ED[:,idx]
    print("### expected energy_MF(t=tau)",ene_MF[0],ene_MF[1])
    print("### expected energy_ED(t=tau)",ene_ED[0],ene_ED[1])
#
    overlap_MF = calc_overlap(psi_MF,vec_MF[:,0])
    overlap_ED = calc_overlap(psi_ED,vec_ED[:,0])
    print("### overlap_MF",overlap_MF)
    print("### overlap_ED",overlap_ED)
    end = time.time()
    print("### time: calculate overlap",end - start)

    if flag_fig_mag == 1:
        print("### list_ED_norm2",list_ED_norm2)
        print("### list_ED_valSx",list_ED_valSx)
        print("### list_ED_valSy",list_ED_valSy)
        print("### list_ED_valSz",list_ED_valSz)
        print("### list_ED_valHam",list_ED_valHam)
#
        fig60 = plt.figure()
        fig60.suptitle("norm^2")
#        plt.plot(list_time,list_MF_norm2,label="inf. range TFI, MF")
        plt.plot(list_time,list_ED_norm2,label="1d n.n. TFI, ED")
        plt.legend(bbox_to_anchor=(0,1),loc='upper left',borderaxespad=1,fontsize=12)
        plt.xlabel("time (from 0 to tau)")
        fig60.savefig("fig_opt_norm2.png")
#
        fig70 = plt.figure()
        fig70.suptitle("<S^x>/S")
#        plt.plot(list_time,list_MF_valSx,label="inf. range TFI, MF")
        plt.plot(list_time,list_ED_valSx,label="1d n.n. TFI, ED")
        plt.legend(bbox_to_anchor=(0,0),loc='lower left',borderaxespad=1,fontsize=12)
        plt.xlabel("time (from 0 to tau)")
        fig70.savefig("fig_opt_valSx.png")
#
        fig80 = plt.figure()
        fig80.suptitle("<S^y>/S")
#        plt.plot(list_time,list_MF_valSy,label="inf. range TFI, MF")
        plt.plot(list_time,list_ED_valSy,label="1d n.n. TFI, ED")
        plt.legend(bbox_to_anchor=(0,1),loc='upper left',borderaxespad=1,fontsize=12)
        plt.xlabel("time (from 0 to tau)")
        fig80.savefig("fig_opt_valSy.png")
#
        fig90 = plt.figure()
        fig90.suptitle("<S^z>/S")
#        plt.plot(list_time,list_MF_valSz,label="inf. range TFI, MF")
        plt.plot(list_time,list_ED_valSz,label="1d n.n. TFI, ED")
        plt.legend(bbox_to_anchor=(0,1),loc='upper left',borderaxespad=1,fontsize=12)
        plt.xlabel("time (from 0 to tau)")
        fig90.savefig("fig_opt_valSz.png")
#
        fig100 = plt.figure()
        fig100.suptitle("energy density")
#        plt.plot(list_time,list_MF_valHam,label="inf. range TFI, MF")
##        plt.plot(list_time,list_ED_valHam,label="1d n.n. TFI, ED")
        plt.plot(list_time,np.array(list_ED_valHam)/N,label="1d n.n. TFI, ED")
        plt.legend(bbox_to_anchor=(0,1),loc='upper left',borderaxespad=1,fontsize=12)
        plt.xlabel("time (from 0 to tau)")
        fig100.savefig("fig_opt_energy.png")

    return overlap_ED

## print results
#    start = time.time()
#    print("list_MF_norm2",list_MF_norm2)
#    print("list_MF_valSx",list_MF_valSx)
#    print("list_MF_valSy",list_MF_valSy)
#    print("list_MF_valSz",list_MF_valSz)
#    print("list_MF_valHam",list_MF_valHam)
#    print("list_ED_norm2",list_ED_norm2)
#    print("list_ED_valSx",list_ED_valSx)
#    print("list_ED_valSy",list_ED_valSy)
#    print("list_ED_valSz",list_ED_valSz)
#    print("list_ED_valHam",list_ED_valHam)
#    end = time.time()
#    print("time: print",end - start)

## plot evolution
#    start = time.time()
##
#    fig10 = plt.figure()
#    fig10.suptitle("Gamma")
#    plt.plot(list_time,list_Gamma)
#    plt.xlabel("time (from 0 to tau)")
#    fig10.savefig("fig_Gamma.png")
##
#    fig20 = plt.figure()
#    fig20.suptitle("GammaDot")
#    plt.plot(list_time,list_GammaDot)
#    plt.xlabel("time (from 0 to tau)")
#    fig20.savefig("fig_GammaDot.png")
##
#    fig30 = plt.figure()
#    fig30.suptitle("Mz")
#    plt.plot(list_time,list_Mz)
#    plt.xlabel("time (from 0 to tau)")
#    fig30.savefig("fig_Mz.png")
##
#    fig40 = plt.figure()
#    fig40.suptitle("MzDot")
#    plt.plot(list_time,list_MzDot)
#    plt.xlabel("time (from 0 to tau)")
#    fig40.savefig("fig_MzDot.png")
##
#    fig50 = plt.figure()
#    fig50.suptitle("ThDot")
#    plt.plot(list_time,list_ThDot)
#    plt.xlabel("time (from 0 to tau)")
#    fig50.savefig("fig_ThDot.png")
##
#    fig60 = plt.figure()
#    fig60.suptitle("norm^2")
#    plt.plot(list_time,list_MF_norm2,label="inf. range TFI, MF")
#    plt.plot(list_time,list_ED_norm2,label="1d n.n. TFI, ED")
#    plt.legend(bbox_to_anchor=(0,1),loc='upper left',borderaxespad=1,fontsize=12)
#    plt.xlabel("time (from 0 to tau)")
#    fig60.savefig("fig_norm2.png")
##
#    fig70 = plt.figure()
#    fig70.suptitle("<S^x>/S")
#    plt.plot(list_time,list_MF_valSx,label="inf. range TFI, MF")
#    plt.plot(list_time,list_ED_valSx,label="1d n.n. TFI, ED")
#    plt.legend(bbox_to_anchor=(0,0),loc='lower left',borderaxespad=1,fontsize=12)
#    plt.xlabel("time (from 0 to tau)")
#    fig70.savefig("fig_valSx.png")
##
#    fig80 = plt.figure()
#    fig80.suptitle("<S^y>/S")
#    plt.plot(list_time,list_MF_valSy,label="inf. range TFI, MF")
#    plt.plot(list_time,list_ED_valSy,label="1d n.n. TFI, ED")
#    plt.legend(bbox_to_anchor=(0,1),loc='upper left',borderaxespad=1,fontsize=12)
#    plt.xlabel("time (from 0 to tau)")
#    fig80.savefig("fig_valSy.png")
##
#    fig90 = plt.figure()
#    fig90.suptitle("<S^z>/S")
#    plt.plot(list_time,list_MF_valSz,label="inf. range TFI, MF")
#    plt.plot(list_time,list_ED_valSz,label="1d n.n. TFI, ED")
#    plt.legend(bbox_to_anchor=(0,1),loc='upper left',borderaxespad=1,fontsize=12)
#    plt.xlabel("time (from 0 to tau)")
#    fig90.savefig("fig_valSz.png")
##
#    fig100 = plt.figure()
#    fig100.suptitle("energy density")
#    plt.plot(list_time,list_MF_valHam,label="inf. range TFI, MF")
##    plt.plot(list_time,list_ED_valHam,label="1d n.n. TFI, ED")
#    plt.plot(list_time,np.array(list_ED_valHam)/N,label="1d n.n. TFI, ED")
#    plt.legend(bbox_to_anchor=(0,1),loc='upper left',borderaxespad=1,fontsize=12)
#    plt.xlabel("time (from 0 to tau)")
#    fig100.savefig("fig_energy.png")
##
##    plt.show()
#    end = time.time()
#    print("time: plot",end - start)

#----

def make_curve_spline(coeff,xmin,ymin,xmax,ymax,Nsteps):
    len_c = len(coeff)
    list_x = np.linspace(xmin,xmax,len_c+2)
    list_y = [ymin, ymax]
    list_y[1:1] = coeff
    list_y = np.array(list_y)
    func_spline = interp1d(list_x,list_y,kind="cubic")
    list_x2 = np.linspace(xmin,xmax,Nsteps)
    list_y2 = func_spline(list_x2)
    return list_x2, list_y2

#----

def main():
    np.set_printoptions(threshold=10000)
    args = parse_args()
    N = args.N
#    twoS = args.twoS
    twoS = 1 ## spin 1/2
    tau = args.tau
#
    J = 1.0
    Hz = 1e-3
    Nbond = N
    invS = 2.0/twoS
    dt = tau/1000.0
    time_i = 0.0
    time_f = tau
    Nsteps = int(tau/dt+0.1)+1
    list_time = [time_i+i*(time_f-time_i)/(Nsteps-1) for i in range(Nsteps)]
#---- MF schedule
    list_Gamma, list_GammaDot, list_Mz, list_MzDot, list_ThDot \
        = make_interaction(tau,J,Hz,list_time)
    flag_fig_mag = 0
    overlap_ED = calc_overlap_final_state(N,twoS,tau,J,Hz,Nbond,invS,dt,Nsteps,\
        list_time,list_Gamma,list_ThDot,flag_fig_mag)
#    calc_overlap_final_state(N,twoS,tau,J,Hz,Nbond,invS,dt,Nsteps,\
#        list_time,list_Gamma,list_GammaDot,list_Mz,list_MzDot,list_ThDot,flag_fig_mag)
#
    fig = plt.figure()
    plt.plot(list_time,list_Gamma)
    fig.savefig("fig_curve_MF_Gamma.png")
#
    fig = plt.figure()
    plt.plot(list_time,list_ThDot)
    fig.savefig("fig_curve_MF_ThDot.png")
#---- end of MF schedule
#
#---- bo
### Gamma (coeff of Sx, to be opt.): 2 --> 0
### ThDot (coeff of Sy, to be opt.): 0 --> 0
### J (coeff of Sz, fixed): 1
### Hz (coeff of Sz.Sz, fixed): 1e-3
#
    Gamma_i = 2.0
    Gamma_f = 0.0
    ThDot_i = 0.0
    ThDot_f = 0.0
#    Gamma_coeff = [1.5,1.0]
#    ThDot_coeff = [-0.1,-0.2]
#
#    gmin = 0.0
#    gmax = 2.0
#    tmin = -2.5
#    tmax = 0.0
#
    gmin = -10.0
    gmax = 10.0
    tmin = -10.0
    tmax = 10.0
#
#    gmin = -3.0
#    gmax = 3.0
#    tmin = -3.0
#    tmax = 3.0
#
#    def black_box_function(g0,g1,g2,t0,t1,t2):
#        Gamma_coeff = [g0,g1,g2]
#        ThDot_coeff = [t0,t1,t2]
#    def black_box_function(g0,g1,g2,g3,g4,g5,t0,t1,t2,t3,t4,t5):
#        Gamma_coeff = [g0,g1,g2,g3,g4,g5]
#        ThDot_coeff = [t0,t1,t2,t3,t4,t5]
    def black_box_function(g0,g1,g2,g3,g4,g5,g6,g7,g8,t0,t1,t2,t3,t4,t5,t6,t7,t8):
        Gamma_coeff = [g0,g1,g2,g3,g4,g5,g6,g7,g8]
        ThDot_coeff = [t0,t1,t2,t3,t4,t5,t6,t7,t8]
        list_time_bo, list_Gamma_bo = make_curve_spline(Gamma_coeff,time_i,Gamma_i,time_f,Gamma_f,Nsteps)
        list_time_bo, list_ThDot_bo = make_curve_spline(ThDot_coeff,time_i,ThDot_i,time_f,ThDot_f,Nsteps)
        flag_fig_mag = 0
        overlap_ED = calc_overlap_final_state(N,twoS,tau,J,Hz,Nbond,invS,dt,Nsteps,\
            list_time_bo,list_Gamma_bo,list_ThDot_bo,flag_fig_mag)
        return overlap_ED
#
    pbounds = {\
        'g0':(gmin,gmax),'g1':(gmin,gmax),'g2':(gmin,gmax),\
        'g3':(gmin,gmax),'g4':(gmin,gmax),'g5':(gmin,gmax),\
        'g6':(gmin,gmax),'g7':(gmin,gmax),'g8':(gmin,gmax),\
        't0':(tmin,tmax),'t1':(tmin,tmax),'t2':(tmin,tmax),\
        't3':(tmin,tmax),'t4':(tmin,tmax),'t5':(tmin,tmax),\
        't6':(tmin,tmax),'t7':(tmin,tmax),'t8':(tmin,tmax),\
        }
    bo = BayesianOptimization(
        f=black_box_function,
        pbounds=pbounds,
        random_state=987234,
    )
#    bo.maximize(n_iter=10)
#    bo.maximize(n_iter=1)
#    bo.maximize(n_iter=20,acq="ucb",kappa=10)
    bo.maximize(n_iter=20,acq="ei",xi=1e-1)
#    bo.maximize(n_iter=20,acq="poi",xi=1e-1)
    print("###### bo.max",bo.max)
#
    g0 = bo.max['params']['g0']
    g1 = bo.max['params']['g1']
    g2 = bo.max['params']['g2']
    g3 = bo.max['params']['g3']
    g4 = bo.max['params']['g4']
    g5 = bo.max['params']['g5']
    g6 = bo.max['params']['g6']
    g7 = bo.max['params']['g7']
    g8 = bo.max['params']['g8']
    t0 = bo.max['params']['t0']
    t1 = bo.max['params']['t1']
    t2 = bo.max['params']['t2']
    t3 = bo.max['params']['t3']
    t4 = bo.max['params']['t4']
    t5 = bo.max['params']['t5']
    t6 = bo.max['params']['t6']
    t7 = bo.max['params']['t7']
    t8 = bo.max['params']['t8']
#    Gamma_coeff = [g0,g1,g2]
#    ThDot_coeff = [t0,t1,t2]
#    Gamma_coeff = [g0,g1,g2,g3,g4,g5]
#    ThDot_coeff = [t0,t1,t2,t3,t4,t5]
    Gamma_coeff = [g0,g1,g2,g3,g4,g5,g6,g7,g8]
    ThDot_coeff = [t0,t1,t2,t3,t4,t5,t6,t7,t8]
    print("###### Gamma_coeff,ThDot_coeff",Gamma_coeff,ThDot_coeff)
    list_time_bo, list_Gamma_bo = make_curve_spline(Gamma_coeff,time_i,Gamma_i,time_f,Gamma_f,Nsteps)
    list_time_bo, list_ThDot_bo = make_curve_spline(ThDot_coeff,time_i,ThDot_i,time_f,ThDot_f,Nsteps)
    flag_fig_mag = 1
    overlap_ED = calc_overlap_final_state(N,twoS,tau,J,Hz,Nbond,invS,dt,Nsteps,\
        list_time_bo,list_Gamma_bo,list_ThDot_bo,flag_fig_mag)
    print("###### overlap_ED",overlap_ED)
#
    fig = plt.figure()
    plt.plot(list_time_bo,list_Gamma_bo)
    fig.savefig("fig_curve_opt_Gamma.png")
#
    fig = plt.figure()
    plt.plot(list_time_bo,list_ThDot_bo)
    fig.savefig("fig_curve_opt_ThDot.png")
#---- end of bo

if __name__ == "__main__":
    main()
