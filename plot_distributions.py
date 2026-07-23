# -*- coding: utf-8 -*-
"""
Created on Tue May 30 09:09:36 2023

@author: vanio
"""
import matplotlib.pyplot as plt
import numpy as np
#from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf, acovf
from math import isnan
#from   qHMM import qOperation

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def plotDistributionsQO(qOperation,samples,targetDist, Title, cur_iter=0, plot_window = 50, fSaveName=None):           
    currentDist = qOperation.getWordSetProb(samples, qOperation.get_pStartState() )
    #targetDist
    #samples
    # set width of bar
    barWidth = 0.3
    edgeWidth = 0.025
    
    fig = plt.subplots(figsize =(8, 2))
    data_window = len(samples) 
    #plt. figsize(8,4)

    for w in range(int(data_window/plot_window)):
        wstr = w*plot_window
        wend = wstr+plot_window
        
        # Set position of bar on X axis
        br1 = np.arange(len(samples[wstr:wend]))
        br2 = [x + (1.1)*barWidth for x in br1]

        # Plot
        #plt.bar(br1, [y[1] for y in currentDist[wstr:wend]], color ='r', edgecolor='grey', linewidth= edgeWidth,  width = barWidth,   label = 'Model')
        #plt.bar(br2, [y[1] for y in targetDist[wstr:wend]],  color ='g', edgecolor='grey', linewidth= edgeWidth, width = barWidth,   label = 'Target')
        plt.bar(br1, [y[1] for y in currentDist[wstr:wend]], color ='r',  width = barWidth,   label = 'Model')
        plt.bar(br2, [y[1] for y in targetDist[wstr:wend]],  color ='b',  width = barWidth,   label = 'Target')

    
        # Adding Xticks
        plt.xlabel('Sequence',  fontsize = 8)
        plt.ylabel('Probability', fontsize = 8)
        plt.xticks([r + barWidth for r in range(len(samples[wstr:wend]))], samples[wstr:wend])
        plt.xticks(rotation='vertical', fontsize = 7)
        if Title != None:
            plt.title(Title+'['+str(wstr)+'-'+str(wend)+']')
        
        plt.legend()
        
        if fSaveName != None:
            #plt. figsize(8,4)
            plt.savefig(fSaveName+str(w)+".pdf", format="pdf", bbox_inches="tight")
        
        plt.show()
        
    wstr = int(data_window/plot_window)*plot_window    
    wend = data_window
    
    # Set position of bar on X axis
    br1 = np.arange(len(samples[wstr:wend]))
    br2 = [x + (2)*barWidth for x in br1]

    # Plot
    plt.bar(br1, [y[1] for y in currentDist[wstr:wend]], color ='r', edgecolor='grey', linewidth= edgeWidth, width = barWidth,   label = 'Model')
    plt.bar(br2, [y[1] for y in targetDist[wstr:wend]],  color ='g', edgecolor='grey', linewidth= edgeWidth, width = barWidth,   label = 'Target')


    # Adding Xticks
    plt.xlabel('Sequence',  fontsize = 14)
    plt.ylabel('Probability', fontsize = 14)
    plt.xticks([r + barWidth for r in range(len(samples[wstr:wend]))], samples[wstr:wend])
    plt.xticks(rotation='vertical',  fontsize = 7)
    if Title != None:
        plt.title(Title+'['+str(wstr)+'-'+str(wend)+']') 
    plt.legend()    
    plt.show()


#------------------------------------------------------------------------------

#-----------------------------------------------------------------------------
def plotDistributions(Dist1, Dist2, samples,  Title, Label1, Label2, c1 = 'red', c2='green' ):           
    #targetDist
    #samples
    # set width of bar
    barWidth = 0.2
    #fig = plt.subplots(figsize =(12, 8))
    data_window = len(Dist2) 
    plot_window = 62 
    if plot_window > data_window:
        plot_window=data_window
     

    for w in range(min(int(data_window/plot_window),5)):
        wstr = w*plot_window
        wend = wstr+plot_window
        
        # Set position of bar on X axis
        br1 = np.arange(len(samples[wstr:wend]))
        br2 = [x + barWidth for x in br1]

        # Plot
        plt.bar(br1, [y for y in Dist1[wstr:wend]],  color =c1, width = barWidth,  label = Label1)
        plt.bar(br2, [y for y in Dist2[wstr:wend]],  color =c2, width = barWidth,   label = Label2)
    
    
        # Adding Xticks
        plt.xlabel('sequence',  fontsize = 5)
        plt.ylabel('probability', fontsize = 5)
        plt.xticks([r + barWidth for r in range(len(samples[wstr:wend]))], samples[wstr:wend])
        plt.xticks(rotation='vertical', fontsize = 7)
        if len(Title)>0: 
            plt.title(Title+'['+str(wstr)+'-'+str(wend)+']')
        
        plt.legend()    
        plt.show()
        
    if wend < data_window:
        wstr = int(data_window/plot_window)*plot_window    
        wend = data_window
        
        # Set position of bar on X axis
        br1 = np.arange(len(samples[wstr:wend]))
        br2 = [x + barWidth for x in br1]
    
        # Plot
        plt.bar(br1, [y for y in Dist1 [wstr:wend]], color =c1, width = barWidth,   label = Label1)
        plt.bar(br2, [y for y in Dist2[wstr:wend]],  color =c2, width = barWidth,   label = Label2)
    
    
        # Adding Xticks
        plt.xlabel('Sequences',  fontsize = 7)
        plt.ylabel('Probabilities', fontsize = 7)
        plt.xticks([r  for r in range(len(samples[wstr:wend]))], samples[wstr:wend])
        plt.xticks(rotation='vertical',  fontsize = 7)
        plt.title(Title) 
        #plt.savefig(Title+'.pdf')    
        plt.show()
#-----------------------------------------------------------------------------
#------------------------------------------------------------------------------
def plotDistributions3(Dist1, Dist2, Dist3, samples,  Title, Label1, Label2, Label3, c1 = 'red', c2='green', c3 = 'blue' ):           
    #targetDist
    #samples
    # set width of bar
    barWidth = 0.2
    #fig = plt.subplots(figsize =(12, 8))
    data_window = len(Dist2) 
    plot_window = 30 

    for w in range(int(data_window/plot_window)):
        wstr = w*plot_window
        wend = wstr+plot_window
        
        # Set position of bar on X axis
        br1 = np.arange(len(samples[wstr:wend]))
        br2 = [x + barWidth for x in br1]
        br3 = [x + 2*barWidth for x in br1]

        # Plot
        #plt.bar(br1, [y[1] for y in Dist1[wstr:wend]],  color =c1, width = barWidth,  label = Label1)
        #plt.bar(br2, [y[1] for y in Dist2[wstr:wend]],  color =c2, width = barWidth,   label = Label2)
        #plt.bar(br3, [y[1] for y in Dist3[wstr:wend]],  color =c3, width = barWidth,   label = Label3)
        plt.bar(br1, Dist1[wstr:wend],  color =c1, width = barWidth,  label = Label1)
        plt.bar(br2, Dist2[wstr:wend],  color =c2, width = barWidth,   label = Label2)
        plt.bar(br3, Dist3[wstr:wend],  color =c3, width = barWidth,   label = Label3)
    
    
        # Adding Xticks
        plt.xlabel('sequence',  fontsize = 5)
        plt.ylabel('probability', fontsize = 5)
        plt.xticks([r + barWidth for r in range(len(samples[wstr:wend]))], samples[wstr:wend])
        plt.xticks(rotation='vertical', fontsize = 7)
        if len(Title)>0: 
            plt.title(Title+'['+str(wstr)+'-'+str(wend)+']')
        
        plt.legend()    
        plt.show()
        
    wstr = int(data_window/plot_window)*plot_window    
    wend = data_window
    
    # Set position of bar on X axis
    br1 = np.arange(len(samples[wstr:wend]))
    br2 = [x + barWidth for x in br1]
    br3 = [x + 2*barWidth for x in br1]
    # Plot
    #plt.bar(br1, [y[1] for y in Dist1 [wstr:wend]], color =c1, width = barWidth,   label = Label1)
    #plt.bar(br2, [y[1] for y in Dist2[wstr:wend]],  color =c2, width = barWidth,   label = Label2)
    #plt.bar(br3, [y[1] for y in Dist3[wstr:wend]],  color =c3, width = barWidth,   label = Label3)
    plt.bar(br1, Dist1[wstr:wend],  color =c1, width = barWidth,   label = Label1)
    plt.bar(br2, Dist2[wstr:wend],  color =c2, width = barWidth,   label = Label2)
    plt.bar(br3, Dist3[wstr:wend],  color =c3, width = barWidth,   label = Label3)

    # Adding Xticks
    plt.xlabel('Sequences',  fontsize = 7)
    plt.ylabel('Probabilities', fontsize = 7)
    plt.xticks([r  for r in range(len(samples[wstr:wend]))], samples[wstr:wend])
    plt.xticks(rotation='vertical',  fontsize = 7)
    if len(Title)>0: 
        plt.title(Title) 
    plt.legend()
    plt.savefig(Title+'.pdf')    
    plt.show()

#-----------------------------------------------------------------------------

def plotDistribution(Dist1, samples,  Title, Label1,  c1 = 'red', fName="bars"):           
    #targetDist
    #samples
    # set width of bar
    barWidth = 0.4
    #fig = plt.subplots(figsize =(12, 8))

    
    # Set position of bar on X axis
    br1 = np.arange(len(samples))
    br2 = [x + barWidth for x in br1]

    # Plot
    plt.bar(br1,  Dist1 , align='center', color =c1, width = barWidth,   label = Label1)
    

    # Adding Xticks
    plt.xlabel('Sequences',  fontsize = 7)
    plt.ylabel('Probabilities', fontsize = 7)
    plt.xticks([r  for r in range(len(samples))], samples)
    plt.xticks(rotation='vertical',  fontsize = 5)
    if len(Title)>0: 
        fontsize = 10
        if len(Title)>30:
            fontsize = 6
        plt.title(Title, fontsize=fontsize)
    plt.legend() 
    #plt.savefig(fName+'.pdf')
    plt.show()
    
    
    
  
 
    
#------------------------------------------------------------------------------
def plot_acf(acf, confint, title=''):
    lag = np.arange(len(acf))  # Lag values
    acf = [1 if isnan(v) else v for v in acf]
    # Plot autocorrelation function
    #plt.stem(lag, acf, basefmt=" ", use_line_collection=False, markerfmt='bo', linefmt='b-')
    plt.scatter(lag, acf, color='b', marker ='o', s=5)
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.title('ACF '+title)

    # Plot confidence intervals
    label='Confidence Intervals'
    plt.fill_between(lag, confint[:, 0], confint[:, 1], color='red', alpha=0.2 )

    if min(acf) > 0.0:    
        plt.ylim(ymin=-0.2)
    else:
        plt.ylim(ymin=1.5*min(acf))
    # Show plot
    #plt.legend()
    plt.grid(True)
    plt.show()
   
#------------------------------------------------------------------------------

def analyze_acf(date, Y,   security, data_type, series_type):
    fft = True
    nlags = 50 
    alpha = 0.05 # conf interval
    # Autocorrelation for Y1   
    
    title=    ' '+data_type+' '+ series_type+ security + date
    acf_mp, confint =acf(Y,  nlags=nlags, qstat=False, fft=fft, alpha=alpha)
    plot_acf(acf_mp, confint, title)


