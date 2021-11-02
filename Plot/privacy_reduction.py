# -*- coding: utf-8 -*-

"""
note something here

"""

__author__ = 'Wang Chen'
__time__ = '2019/7/22'

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoLocator, ScalarFormatter
    import sys
    import os 
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes  
    plt.figure(figsize=(20, 13))
    #plt.title('hitrate (city level)',fontsize=40) 
    #optimal      = list(np.array([0.35486,0.84040171,0.91643094,1         ])*100)
    #random       = list(np.array([3.87249, 3.39743, 3.05268, 2.80303, 2.58978, 2.#42809,
    #   2.29468, 2.16831, 2.05126, 1.93582, 1.83487, 1.7502 , 1.66939,
    #   1.59503, 1.52718, 1.46064, 1.40018, 1.33967, 1.28351, 1.23272,
    #   1.18247, 1.13662, 1.09053, 1.04733, 1.00647, 0.96889, 0.93318,
    #   0.8988 , 0.86455, 0.83082]))
    efficiency   = list(np.array([2.21577, 1.8715 , 1.7486 , 1.66805, 1.60001, 1.54552, 1.49728,
       1.46524, 1.42328, 1.38784, 1.33888, 1.28531, 1.24732, 1.23574]))
    privacy      = list(np.array([1.25203, 1.09144, 1.02947, 1.00001, 0.97631, 0.96008, 0.95195,
       0.94009, 0.92608, 0.91054, 0.88761, 0.8607 , 0.81159, 0.73957]))
    Deepcache    = list(np.array([ 2.06928, 1.65084, 1.47097, 1.37198, 1.29012, 1.23405, 1.19906,
       1.15923, 1.12433, 1.08759, 1.04539, 1.00383, 0.94799, 0.89549,]))
    POC = list(np.array([ 2.36876, 1.88524, 1.65865, 1.52679, 1.42193, 1.35381, 1.31732,
       1.27764, 1.2421 , 1.20925, 1.17299, 1.14005, 1.10341, 1.0725]))
    noneCache    = list(np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]))
    balance = list(np.array([3.5753 , 2.66482, 2.24786, 1.99493, 1.74118, 1.56149, 1.44256,
       1.33242, 1.23067, 1.14405, 1.06181, 1.0016 , 0.96057, 0.9474]))

    x = [10000*n for n in range(1,len(balance))]
    x.append(132689)
    font1 = {
    'weight' : 'bold',
    'size'   : 48,
    }
    plt.xlabel('Request times', font1,position = (0.5,0.0))
    plt.ylabel('Privacy reduction', font1)
   
    plt.ylim(ymin=0, ymax=4)
    plt.xlim(xmin=x[0]-1000,xmax =x[-1]+1000) 
    #fig.align_labels()
    ax = plt.gca()
    #ax.set_xscale('log')   
    ax.yaxis.set_ticks_position('right')
    plt.minorticks_off()
    #ax.spines['right'].set_color('none')
    #names = ['10000','20000','30000','40000','50000','60000','70000','80000','']
    #names = ['','1%(300)','','','','','5%(1500)','','','10%(3000)','','','20%(6000)','','50%(15000)']   
    #ax.xaxis.set_major_locator(AutoLocator())
    #ax.xaxis.set_major_formatter(ScalarFormatter())
    #plt.minorticks_on()
    #plt.xticks([], [])
    plt.tick_params(which='both',direction='in')
    plt.tick_params(which='major',width=5,length=8)
    plt.tick_params(labelsize=40)
    #ax.set_xticklabels([])
    
    #plt.xticks(x, size = 40,rotation=0,verticalalignment ='bottom',position = (0,-0.06))
    plt.yticks(np.arange(0.5, 4.1, 0.5), size = 40)
    
    plt.subplots_adjust(bottom=0.115,right=0.930,left=0.070,top=0.980)

   
    plt.plot(x, privacy, '#ff93ff', label='Privacy-only',    linewidth=5, marker='o', markersize=7, )
    plt.plot(x, efficiency,'#82f479' , label='Efficiency-only', linewidth=5, marker='v', markersize=7, )
    plt.plot(x, balance, '#0edef1', label='Balance',      linewidth=5, marker='*', markersize=7, markerfacecolor = 'none')
    plt.plot(x, noneCache, '#ffcc60', label='None',        linewidth=5, marker='d', markersize=7, )
    #plt.plot(x, random, '#87bbea', label='Random',        linewidth=5, marker='X', markersize=7, ) 
    #plt.plot(x, POC, '#a4a4a4', label='POC',  linewidth=5, marker="P", markersize=7, )
    plt.plot(x, Deepcache, '#ff8c80', label='DPC',  linewidth=5, marker="s", markersize=7, )
    #plt.plot(x, POC, '#a4a4a4', label='POC',linewidth=8, marker='s', markersize=25, markerfacecolor = 'none',linestyle = '-.')
    
    font2 = {
    'weight' : 'normal',
    'size'   : 36,
    }
    handles, labels = ax.get_legend_handles_labels() 
    plt.legend(handles[::-1], labels[::-1],prop = font2, loc='best', ncol=2, framealpha=1,frameon=True,facecolor ='w',columnspacing=0.8,)   

    plt.savefig('/home/ubuntu/data/PEC/Plot/output/privacy_reduction.eps')
    plt.savefig('/home/ubuntu/data/PEC/Plot/output/privacy_reduction_1017.eps')
    plt.savefig('/home/ubuntu/data/PEC/Plot/output/privacy_reduction_1017.jpeg')
    plt.show()
