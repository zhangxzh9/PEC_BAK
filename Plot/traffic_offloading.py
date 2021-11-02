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
    #random       = list(np.array([0.09059, 0.0934 , 0.08723, 0.08825, 0.08818, 0.#08638,  0.08448, 0.08411, 0.08362, 0.08257, 0.08212, 0.08162, 0.08092,
    #    0.08119, 0.08095, 0.08079, 0.08082, 0.08106, 0.08078, 0.08094,
    #    0.08124, 0.08088, 0.08085, 0.0807 , 0.08114, 0.08093, 0.08072,
    #    0.08097, 0.08117, 0.08118])*100)
    efficiency   = list(np.array([0.3012 , 0.30055, 0.3354 , 0.3458 , 0.3687 , 0.37268, 0.37229, 0.37186, 0.37833, 0.37287, 0.37403, 0.37053, 0.36254, 0.3623])*100)
    privacy      = list(np.array([0.3001 , 0.26865, 0.2771 , 0.26515, 0.2704 , 0.26173, 0.25447,
        0.24841, 0.25256, 0.24869, 0.25511, 0.25506, 0.24922, 0.24959])*100)
    Deepcache    = list(np.array([0.2803 , 0.25715, 0.26757, 0.25718, 0.26352, 0.25548, 0.2488 ,
        0.24299, 0.24738, 0.24378, 0.25036, 0.25031, 0.2446 , 0.24499,])*100)
    POC = list(np.array([0.2776 , 0.25695, 0.26843, 0.25783, 0.26438, 0.2564 , 0.24971,
        0.24395, 0.24819, 0.24467, 0.25136, 0.25134, 0.24566, 0.24603,])*100)
    noneCache    = list(np.array([0.3055 , 0.2746 , 0.28223, 0.27058, 0.276  , 0.26712, 0.26043,
        0.25418, 0.25832, 0.25417, 0.26074, 0.26049, 0.25458, 0.25491])*100)
    balance = list(np.array([0.2021 , 0.22355, 0.26893, 0.28963, 0.31518, 0.32228, 0.32414, 0.32591, 0.33854, 0.34012, 0.35338, 0.35732, 0.35405, 0.35499])*100)

    x = [10000*n for n in range(1,len(balance))]
    x.append(132689)
    font1 = {
    'weight' : 'bold',
    'size'   : 48,
    }
    plt.xlabel('Request times', font1,position = (0.5,0.0))
    plt.ylabel('Traffic offloading', font1)
   
    plt.ylim(ymin=20, ymax=41)
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
    
    #plt.xticks(x, size = 10,rotation=0,verticalalignment ='bottom',position = (0,-0.06))
    plt.yticks(np.arange(20, 45, 5),['20%','25%','30%','35%','40%',], size = 40,)
    
    plt.subplots_adjust(bottom=0.115,right=0.930,left=0.070,top=0.980)

   
    plt.plot(x, privacy, '#ff93ff', label='Privacy-only',    linewidth=5, marker='o', markersize=5, )
    plt.plot(x, efficiency,'#82f479' , label='Efficiency-only', linewidth=5, marker='v', markersize=5, )
    plt.plot(x, balance, '#0edef1', label='Balance',      linewidth=5, marker='*', markersize=5, markerfacecolor = 'none')
    plt.plot(x, noneCache, '#ffcc60', label='None',        linewidth=5, marker='d', markersize=5, )
    #plt.plot(x, random, '#87bbea', label='Random',        linewidth=5, marker='X', markersize=5, ) 
    #plt.plot(x, POC, '#a4a4a4', label='POC',  linewidth=5, marker="P", markersize=5, )
    plt.plot(x, Deepcache, '#ff8c80', label='DPC',  linewidth=5, marker="s", markersize=5, )
    #plt.plot(x, POC, '#a4a4a4', label='POC',linewidth=8, marker='s', markersize=25, markerfacecolor = 'none',linestyle = '-.')
    
    
    font2 = {
    'weight' : 'normal',
    'size'   : 28,
    }
    handles, labels = ax.get_legend_handles_labels() 
    plt.legend(handles[::-1], labels[::-1],prop = font2, loc='best', ncol=5, framealpha=1,frameon=True,facecolor ='w',columnspacing=0.8,)   

    plt.savefig('/home/ubuntu/data/PEC/Plot/output/traffic_offloading.eps')
    plt.savefig('/home/ubuntu/data/PEC/Plot/output/traffic_offloading_1017.eps')
    plt.savefig('/home/ubuntu/data/PEC/Plot/output/traffic_offloading_1017.jpeg')
    plt.show()
