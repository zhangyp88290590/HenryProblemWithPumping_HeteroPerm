# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 19:01:44 2020

@author: zhang
"""

import os
import sys
import numpy as np
# Post-process the results
import flopy.utils.binaryfile as bf
# Make the plot
import matplotlib.pyplot as plt
import csv

# run installed version of flopy or add local path
try:
    import flopy
except:
    fpth = os.path.abspath(os.path.join('..', '..'))
    sys.path.append(fpth)
    import flopy

print(sys.version)
print('numpy version: {}'.format(np.__version__))
print('flopy version: {}'.format(flopy.__version__))


workspace = os.path.join('data')
#make sure workspace directory exists
if not os.path.exists(workspace):
    os.makedirs(workspace)

# Input variables for the Henry Problem
Lx = 200
Lz = 100
nlay = 50#250
nrow = 1
ncol = 100 #500
delr = Lx / ncol
delc = 1.0
delv = Lz / nlay
henry_top = 100
henry_botm = np.linspace(henry_top - delv, 0., nlay)
qinflow = 5.702*100*2#m3/day 5.702
dmcoef = 1.62925#1.62925#0.57024 #m2/day  Could also try 1.62925 as another case of the Henry problem
#hk_init = np.load('hk_init864_0p5_5reals.npy')
hk_init = np.load('hk_init86p4_0p5_5range_10reals_new.npy')


hk = np.zeros((nlay,nrow,ncol))

#hk_c = 86.40  #m/day
#alpha_L_arr = [0.0]#, 0.5, 1.0, 2.0, 10.0]
alpha_L = 0.0
#pwrate = [0.0, -0.5, -1.0, -2.5, -5.0, -7.5, -10.0, -12.5, -15.0, -20.0, -25.0, -30.0, -35.0, -40.0, -45.0,-50.0,-55.0] # m^3/day
pwrate = [0.0, -2.5, -5.0, -7.5, -15.0]

pwrate = -100*5.702*2*2
#pwrate = np.linspace(0,5,num=6)
#pwrate = -100*5.702*2*np.array(pwrate)

# create hk field decreasing towards sea
#hk = np.zeros((nlay,nrow,ncol))

#hk[:,:,:] = hk_c
    

sampling_method = {}
toe_x_sample = {}
# sampling strategy
# sampling method keep the old points from previous method:
# 1) [0, 5]
# 2) [0, 3, 5]
# 3) [0, 2, 3, 5]
# 4) [0, 2, 3, 4, 5]
# 5) [0, 1, 2, 3, 4, 5]    

# original pumping rate set up    
# =  {0:[0,5], 1:[0,3,5], 2:[0,2,3,5], 3:[0,2,3,4,5], 4:[0,1,2,3,4,5]}

# low pumping rates to refine the range between 0 and 2

#pwrate = [0.0,0.5,0.75,1.0,1.25,1.5,2.0,3.0,4.0,5.0] this is refined pumping 
# rate with more sampling points from 0 to 2
pwrate_s = {0:[0,2], 1:[0,1,2], 2:[0,0.5,1,2], 
            3:[0.0,0.5,0.75,1.0,1.25,1.5,2.0,3.0,4.0,5.0]}

for n in range(4):
    toe_x_sample[n] = np.zeros(n+2)
    sampling_method[n] = -100*5.702*2*np.array(pwrate_s[n])


    
# #for vertical gradual change of Hk
# hk_n = np.linspace(864,86.4,20)

# for i in range(np.shape(hk_n)[0]):
# #    hk[:,0,0+i*10:10+i*11] = hk_n[i]
#      hk[:,0,0+i*5:5+i*5] = hk_n[i] 
# create hk field increasing towards sea
#hk = hk[:,:,::-1]

# for horizontal gradual change of Hk
# hk_n2 = np.linspace(864,86.4,10)

# for i in range(np.shape(hk_n2)[0]):
# #    hk[:,0,0+i*10:10+i*11] = hk_n[i]
#     hk[0+i*5:5+i*5,0,:] = hk_n2[i]
# hk = hk[::-1,:,:]

# Create 2 stress periods for well package
nper = 2
perlen = [30, 30]
nstp = [15, 15]
steady = [False, False] # SEAWAT requirement see Manual for details

for f in range((np.shape(hk_init)[0])): # range loop
    print(f)
    
    HRspace = os.path.join('HR%s' % f)
    if not os.path.exists(HRspace):
        os.makedirs(HRspace)  

    
    for j in range((np.shape(hk_init)[2])): # realization loop
        print(j)
        #alpha_L = alpha_L_arr[j]
    
        hk_read = hk_init[f,:,j].reshape(50,100)
        hk_ud= np.flipud(hk_read)
        hk[:,0,:] = hk_ud
        
        # YPZconvert hk value to natural log (exp base)
        loghk = np.log(hk) 
    
        hkspace = os.path.join(HRspace, 'hk%s' % j)
        
        if not os.path.exists(hkspace):
            os.makedirs(hkspace)  
    
 
        # sampling method loop 
#        for l in range(len(sampling_method)):
#            print(l)
# dent is not right here
        l=3
        pwrate = sampling_method[l]
            
            
        toe_C= np.zeros(len(pwrate)) # dimensionlize concentration of toe
        toe_x= np.zeros(len(pwrate)) # dimensionlize x position of toe along the bottom

        toe_C17p5= np.zeros(len(pwrate)) # dimensionlize concentration of toe
        toe_x17p5= np.zeros(len(pwrate)) # dimensionlize x position of toe along the bottom

        toe_C1= np.zeros(len(pwrate)) # dimensionlize concentration of toe
        toe_x1= np.zeros(len(pwrate)) # dimensionlize x position of toe along the bottom
    
        samplespace = os.path.join(hkspace,'sample%s' % l)
    
        if not os.path.exists(samplespace):
            os.makedirs(samplespace)  
#23456789
        # pumping rate loop
        for i in range(len(pwrate)): 
            print(i)
            obsspace = os.path.join(samplespace,'pump%s' % i) # combine hk dirc with pump dirc   
            if not os.path.exists(obsspace):
                os.makedirs(obsspace)  
        
            # Create the basic MODFLOW model structure
            modelname = 'henry'
            swt = flopy.seawat.Seawat(modelname, exe_name='swt_v4', model_ws=obsspace)
            print(swt.namefile)

            # save cell fluxes to unit 53
            ipakcb = 53

            # Add DIS package to the MODFLOW model
            # by default steady = True;
            # nper number of stress periods
            # perlen: stress period length; nstp:number of time steps within each stress period
            dis = flopy.modflow.ModflowDis(swt, nlay, nrow, ncol, nper=nper, delr=delr,
                               delc=delc, laycbd=0, top=henry_top,
                               botm=henry_botm, steady = steady,perlen=perlen, nstp=nstp)
            #                               botm=henry_botm, steady = True,perlen=60, nstp=15)

            y, x, z = dis.get_node_coordinates()
            X, Z = np.meshgrid(x, z[:, 0, 0]) 


            # Variables for the BAS package
            ibound = np.ones((nlay, nrow, ncol), dtype=np.int32)
            ibound[:, :, -1] = -1
            bas = flopy.modflow.ModflowBas(swt, ibound, strt=1)
        
            # Add LPF package to the MODFLOW model
            lpf = flopy.modflow.ModflowLpf(swt, hk=hk, vka=hk, ipakcb=ipakcb)

            # Add PCG Package to the MODFLOW model
            pcg = flopy.modflow.ModflowPcg(swt, hclose=1.e-8)
        
            # Add OC package to the MODFLOW model
            # note that the last time step is printed out
            oc = flopy.modflow.ModflowOc(swt, stress_period_data={(1, 14): ['save head', 'save budget']},
                             compact=True)


            # Create WEL and SSM data
            itype = flopy.mt3d.Mt3dSsm.itype_dict()
            wel_data = {}
            ssm_data = {}
            wel_sp1 = []
            ssm_sp1 = []
        
            # parameters for adding a pumping well (location and pumping rate)
            pwlay = 19 # base case 19 [9 14 19 24 29 34]
            pwrow = 0
            pwcol = 49 # base case 49 [4 24 49 74]
        
            # add conc. observation point at the pumping well
            wellobs = [[pwlay,pwrow,pwcol]]
        
        
            for k in range(nlay):
                wel_sp1.append([k, 0, 0, qinflow / nlay])
                ssm_sp1.append([k, 0, 0, 0., itype['WEL']])
                ssm_sp1.append([k, 0, ncol - 1, 35., itype['BAS6']])
        
            #wel_sp2 = wel_sp1+ [[pwlay, pwrow, pwcol, pwrate]]+[[rwlay, rwrow, rwcol, rwrate]]   
            wel_sp2 = wel_sp1+ [[pwlay, pwrow, pwcol, pwrate[i]]]
            wel_data[0] = wel_sp1
            wel_data[1] = wel_sp2

            #ssm_sp2 = ssm_sp1+[[pwlay, pwrow, pwcol, 0.0, itype['WEL']]]+[[rwlay, rwrow, rwcol, 0.0, itype['WEL']]]
            ssm_sp2 = ssm_sp1+[[pwlay, pwrow, pwcol, 0.0, itype['WEL']]]
            ssm_data[0] = ssm_sp1
            ssm_data[1] = ssm_sp2
        
        
        
            wel = flopy.modflow.ModflowWel(swt, stress_period_data=wel_data, ipakcb=ipakcb)

            # Create the basic MT3DMS model structure
            #mt = flopy.mt3d.Mt3dms(modelname, 'nam_mt3dms', mf, model_ws=workspace)
            btn = flopy.mt3d.Mt3dBtn(swt, nprs=-5, prsity=0.35, sconc=35., ifmtcn=0,
                         chkmas=True, obs = wellobs, nprobs=10, nprmas=10, dt0=0.001)
            adv = flopy.mt3d.Mt3dAdv(swt, mixelm=0)
            dsp = flopy.mt3d.Mt3dDsp(swt, al=alpha_L, trpt=1., trpv=1., dmcoef=dmcoef)
            gcg = flopy.mt3d.Mt3dGcg(swt, iter1=500, mxiter=1, isolve=1, cclose=1e-7)
            ssm = flopy.mt3d.Mt3dSsm(swt, stress_period_data=ssm_data)

            # Create the SEAWAT model structure
            #mswt = flopy.seawat.Seawat(modelname, 'nam_swt', mf, mt, model_ws=workspace, exe_name='swtv4')
            vdf = flopy.seawat.SeawatVdf(swt, iwtable=0, densemin=0, densemax=0,
                             denseref=1000., denseslp=0.7143, firstdt=1e-3)


            # Write the input files
            swt.write_input()

            # Try to delete the output files, to prevent accidental use of older files
            try:
                os.remove(os.path.join(obsspace, 'MT3D001.UCN'))
                os.remove(os.path.join(obsspace, modelname + '.hds'))
                os.remove(os.path.join(obsspace, modelname + '.cbc'))
            except:
                pass

            v = swt.run_model(silent=True, report=True)

            for idx in range(-3, 0):
                print(v[1][idx])
        
            figurepath = os.path.join(obsspace,'figures')    
            if not os.path.exists(figurepath):
                os.makedirs(figurepath)    
    
            # # Post-process the results
            # import numpy as np
            # import flopy.utils.binaryfile as bf

            # Load data
            ucnobj = bf.UcnFile(os.path.join(obsspace, 'MT3D001.UCN'), model=swt)
            times = ucnobj.get_times()
            concentration = ucnobj.get_data(totim=times[-1])

#------------------------------------------------------------------------------
            # YPZ find the toe position of the sea water intrusion front

            conc2d = np.ndarray((50,100),np.longlong)
            conc2d = concentration[:,0,:]
        
            for m in range(np.shape(conc2d)[1]):
                if conc2d[49,m] == 35.000000: break # jump out of the loop when finding the 35 seawater conc.
                toe_C[i]= conc2d[49,m-1] # concentration of toe
                toe_x[i] = X[49,m-1] # x position of toe along the bottom
                    
            for m in range(np.shape(conc2d)[1]):
                if conc2d[49,m] == 17.5000000: break # jump out of the loop when finding the 35 seawater conc.
                toe_C17p5[i]= conc2d[49,m-1] # concentration of toe
                toe_x17p5[i] = X[49,m-1] # x position of toe along the bottom
#23456789123456789                
            for m in range(np.shape(conc2d)[1]):
                if conc2d[49,m] == 1.0000000: break # jump out of the loop when finding the 35 seawater conc.
                toe_C1[i]= conc2d[49,m-1] # concentration of toe
                toe_x1[i] = X[49,m-1] # x position of toe along the bottom                    
                

#------------------------------------------------------------------------------

#23456789123456789
            cbbobj = bf.CellBudgetFile(os.path.join(obsspace, 'henry.cbc'))
            times = cbbobj.get_times()
            qx = cbbobj.get_data(text='flow right face', totim=times[-1])[0]
            qy = np.zeros((nlay, nrow, ncol), dtype=np.float)
            qz = cbbobj.get_data(text='flow lower face', totim=times[-1])[0]

            # Average flows to cell centers
            qx_avg = np.empty(qx.shape, dtype=qx.dtype)
            qx_avg[:, :, 1:] = 0.5 * (qx[:, :, 0:ncol-1] + qx[:, :, 1:ncol])
            qx_avg[:, :, 0] = 0.5 * qx[:, :, 0]
            qz_avg = np.empty(qz.shape, dtype=qz.dtype)
            qz_avg[1:, :, :] = 0.5 * (qz[0:nlay-1, :, :] + qz[1:nlay, :, :])
            qz_avg[0, :, :] = 0.5 * qz[0, :, :]



            # fig = plt.figure(figsize=(10, 10))
            # ax = fig.add_subplot(1, 1, 1, aspect='equal')
            # ax.imshow(concentration[:, 0, :], interpolation='nearest',
            #            extent=(0, Lx, 0, Lz))
            # y, x, z = dis.get_node_coordinates()
            # X, Z = np.meshgrid(x, z[:, 0, 0])
            # iskip = 3
            # ax.quiver(X[::iskip, ::iskip], Z[::iskip, ::iskip],
            #            qx_avg[::iskip, 0, ::iskip], -qz_avg[::iskip, 0, ::iskip],
            #            color='w', scale=5, headwidth=3, headlength=2,
            #            headaxislength=2, width=0.0025)
            # plt.savefig(os.path.join(workspace, 'henry.png'))
            # plt.show()
        
            fig = plt.figure(figsize=(12,9))
            ax = fig.add_subplot(1, 1, 1, aspect="equal")
            pmv = flopy.plot.PlotCrossSection(model=swt, ax=ax, line={"row": 0})
            arr = pmv.plot_array(concentration,vmin = 0, vmax=35.0)
            contours = pmv.contour_array(concentration, colors="white")
            ax.clabel(contours, fmt="%2.1f")
            #pmv.plot_vector(qx, qy, -qz, color="white", kstep=3, hstep=3)
            plt.colorbar(arr, shrink=0.5, ax=ax)
            ax.set_title("Simulated Concentrations (g/L)");
            plt.savefig(os.path.join(figurepath,'simConc.png'))
            plt.close(fig)


            # Extract the heads
            fname = os.path.join(obsspace, 'henry.hds')
            headobj = bf.HeadFile(fname)
            times = headobj.get_times()
            head = headobj.get_data(totim=times[-1])

            # Make a simple head plot
            # fig = plt.figure(figsize=(10, 10))
            # ax = fig.add_subplot(1, 1, 1, aspect='equal')
            # im = ax.imshow(head[:, 0, :], interpolation='nearest',
            #                extent=(0, Lx, 0, Lz))
            # ax.set_title('Simulated Heads');

            fig = plt.figure(figsize=(12, 9))
            ax = fig.add_subplot(1, 1, 1, aspect="equal")
            pmv = flopy.plot.PlotCrossSection(model=swt, ax=ax, line={"row": 0})
            arr = pmv.plot_array(head)
            contours = pmv.contour_array(head, colors="white")
            ax.clabel(contours, fmt="%2.2f")
            plt.colorbar(arr, shrink=0.5, ax=ax)
            ax.set_title("Simulated Heads (m)");
            plt.savefig(os.path.join(figurepath,'simHeads.png'))
            plt.close(fig)
    

            y, x, z = dis.get_node_coordinates()
            X, Z = np.meshgrid(x, z[:, 0, 0])
            iskip = 3
            ax.quiver(X[::iskip, ::iskip], Z[::iskip, ::iskip],
                          qx_avg[::iskip, 0, ::iskip], -qz_avg[::iskip, 0, ::iskip],
                          color='w', scale=5, headwidth=3, headlength=2,
                          headaxislength=2, width=0.0025)



            fig = plt.figure(figsize=(12, 9))
            ax = fig.add_subplot(1, 1, 1, aspect="equal")
            pmv = flopy.plot.PlotCrossSection(model=swt, ax=ax, line={"row": 0})
            arr = pmv.plot_array(hk)
            contours = pmv.contour_array(hk, colors="white")
            plt.colorbar(arr, shrink=0.5, ax=ax)

            pmv = flopy.plot.PlotCrossSection(model=swt, line={"row": 0})
            linecollection = pmv.plot_grid(linewidth=0.45)
            ax.set_title("Log (Hydraulic Conductivity) (m/day)");
            plt.savefig(os.path.join(figurepath,'Log_HkValue.png'))
            plt.close(fig)
            #plt.show()

            # convert to numpy array 
            b = np.array(toe_x).reshape(-1,1)
            # write out positions of the sea front toe to each sampling folder
            filepath = os.path.join(samplespace,'toe_x.csv')
            with open(filepath, 'w', newline='') as file:
                mywriter = csv.writer(file, delimiter=' ')
                mywriter.writerows(b)
#23456789
        # convert to numpy array 
        b2 = np.array(toe_x17p5).reshape(-1,1)
        # write out positions of the sea front toe to each sampling folder
        filepath = os.path.join(samplespace,'toe_x17p5.csv')
        with open(filepath, 'w', newline='') as file:
            mywriter = csv.writer(file, delimiter=' ')
            mywriter.writerows(b2)
                
        # convert to numpy array 
        b3 = np.array(toe_x1).reshape(-1,1)
        # write out positions of the sea front toe to each sampling folder
        filepath = os.path.join(samplespace,'toe_x1.csv')
        with open(filepath, 'w', newline='') as file:
            mywriter = csv.writer(file, delimiter=' ')
            mywriter.writerows(b2) 