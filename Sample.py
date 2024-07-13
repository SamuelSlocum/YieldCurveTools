#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 23:55:59 2024

@author: samuelslocum
"""

from ACMTools import *


print()
print("   ACM Yield Curve Model")
print("===========================")
print("Checking Requirements")

if sys.version_info[0] < 3:
    print("You are not using Python 3")
    print("WARNING: This code was tested with Python 3.6.1")
   
#%%#------- SECTION 1: Parameter Specification and Pulling Raw Data

#Where to get the data
DATA_LINK = 'https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv'

#Number of months of history to use in estimating the model
T = 758

#These are the tenors used to fit the model
MONTHS = [6*(x+1) for x in range(20)]

#Derived parameters
YEARS_OUTPUT = [x/12 for x in range(1,121)]
YEARS = [x/12 for x in MONTHS]
N = len(MONTHS)

#Pull the Fed's fitted Svennson yield curve parameters. 
print("Fetching yield curve data...")
monthly_data = pull_yc_data(DATA_LINK)

#%%#------- SECTION 2: Model Estimation

#First generate yield curves
yield_df = Svensson(MONTHS,monthly_data)

#Next, compute theoretic bond returns based off of yield curve, put them in the proper format.
RX, Y, RF = fit_yields_returns(MONTHS, monthly_data, T)

#Fit an ACM model
#For best use, keep K>3
K = 5
ACM = AMCModel(RX,Y,RF,K)
ACM.fit()

#The user can input month horizons and short rate values, and the model will filter forward and produce a yield curve conditional on these estimates.
proj_horizon = [6,12,24]
target = [0.045,0.035,0.02]

#These are the curves conditional on user provided short-rate values and horizons. They are plotted below
curve_projection = ACM.walk_forward(proj_horizon,target)

#%%#------- SECTION 3.1: Plotting Term Premia

print()
print("Generating Plots")

#Lay out the figure
font = {'family' : 'Times New Roman',
        'size'   : 17}
plt.rc('font', **font)

f, (ax1, ax2) = plt.subplots(1, 2,figsize=(12,6))

#PLOT 1: Summary of the most recent yield curve.

#Plotting data
date_final = yield_df.index[-1]
ax1.plot(YEARS_OUTPUT,ACM.yields_fit[:,-1],color='b',label='Model-Fitted Yields')
ax1.plot(YEARS,Y[:,-1],color=[.1,.1,.1],marker='+',markersize=10,linestyle='none',label='Fed-Provided Yields')
ax1.plot(YEARS_OUTPUT,ACM.yields_fit_nolambda[:,-1],color=[.2,.6,.2],label='Expected Average Short-Rate')


#Labelling/Formatting
ax1.yaxis.grid()
vals = ax1.get_yticks()
ax1.set_yticklabels(['{:,.1%}'.format(x) for x in vals])
ax1.set_title("Yield Curve: "+date_final.strftime("%B %d, %Y"))
ax1.set_xlabel("Maturity(years)")
ax1.set_ylabel("Yield(Continuously Compounded %)")
ax1.legend()

#PLOT 2: History of term premiums

#Plotting data
x_axis2 = yield_df.index[-(T+1):]
ax2.plot(x_axis2,ACM.yields_fit[-1,:]-ACM.yields_fit_nolambda[-1,:],color='k',label='10y Term-Premium')
ax2.plot(x_axis2,ACM.yields_fit[-1,:],color='b',label='10y Model-Fitted yield')
ax2.plot(x_axis2,ACM.yields_fit_nolambda[-1,:],color=[.2,.6,.2],label='10y Expected Average Short-Rate')

#Labelling/Formatting
ax2.set_xlabel("Date")
ax2.yaxis.grid()
vals = ax2.get_yticks()
ax2.set_yticklabels(['{:,.1%}'.format(x) for x in vals])
ax2.set_title("10y Term-Premium History")
ax2.legend()

graph_filename = "ACMmodel_graphs_baseline_"+date.today().strftime("%Y%m%d")
print("Saving graphs as "+graph_filename+".png")
plt.savefig(graph_filename)
plt.show()

#%%#------- SECTION 3.2 Plotting Scenarios

print("Generating Plots")

font = {'family' : 'Times New Roman',
        'size'   : 17}
plt.rc('font', **font)

#Lay out the figure
string1 = [0,1,2,3,4,5,6]
alpha=0.75

string2 = ["Bottom" for x in range(len(string1))]
f, ax_list = plt.subplots(1,3,figsize=(6*len(proj_horizon),12))#[string1[:len(proj_horizon)],string2[:len(proj_horizon)]],figsize=(6*len(proj_horizon),12))
#f, ax_list = plt.subplots(1, len(proj_horizon))
f.suptitle("Yield Curve Scenarios as of "+date_final.strftime("%B %d, %Y"), fontsize=30)
#PLOT 1: Summary of the most recent yield curve.
for k in range(len(proj_horizon)):
    #Plotting data
    horizon = proj_horizon[k]
    ax_list[k].plot(YEARS_OUTPUT[3:],ACM.yields_fit[3:,-1],color='b',label='Current Yields')
    ax_list[k].plot(YEARS,Y[:,-1],color=[.1,.1,.1],marker='+',markersize=10,linestyle='none',label='Fed-Provided Yields')
    ax_list[k].plot(YEARS_OUTPUT[3:],curve_projection[horizon]["expected"][3:],color=[.6,.2,.6],label="Expected Curve")
    ax_list[k].fill_between(YEARS_OUTPUT[3:], curve_projection[horizon]["low"][3:], curve_projection[horizon]["high"][3:],color='0.85',label=str(int(100-2*100*alpha))+"% Confidence Interval")
    
    #Labelling/Formatting
    ax_list[k].yaxis.grid()
    vals = ax_list[k].get_yticks()
    ax_list[k].set_yticklabels(['{:,.1%}'.format(x) for x in vals])
    ax_list[k].set_title(str(horizon)+" Months Expected Curve. FFR = "+str(round(target[k]*100,1))+"%")
    #ax_list[k].set_title("Yield Curve: "+date_final.strftime("%B %d, %Y"))
    ax_list[k].set_xlabel("Maturity(years)")
    ax_list[k].yaxis.set_tick_params(which='both', labelbottom=True)
    ax_list[k].sharey(ax_list[2])
    
    if k==0:
        ax_list[k].set_ylabel("Yield (Continuously Compounded %)")
        ax_list[k].legend(loc="lower left")

graph_filename = "ACMmodel_graphs_scenarios_"+date.today().strftime("%Y%m%d")
print("Saving graphs as "+graph_filename+".png")
plt.savefig(graph_filename)

plt.show()




