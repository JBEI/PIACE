#!/usr/local/bin/python3

#Numerics
import random
import numpy as np
import statistics

#Symbolic Algebra
from sympy import *
from sympy.utilities.lambdify import lambdastr

#Optimization Algorithms
from scipy.optimize import differential_evolution
from scipy.integrate import ode,odeint

#Import Plotting Tools
import matplotlib.pyplot as plt
import matplotlib.colors as colors

#Machine Learning Tools
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.decomposition import PCA

from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler,MinMaxScaler,maxabs_scale
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.cross_decomposition import PLSRegression,PLSCanonical
from sklearn.pipeline import Pipeline
#from sklearn import cross_validation

from mpl_toolkits.axes_grid1 import AxesGrid
from mpl_toolkits.axes_grid1 import make_axes_locatable

#=============================================================#
# Utilities                                                   #
#-------------------------------------------------------------#
def addMeasurementNoise(measurements,noise_vector):
    if isinstance(noise_vector, float):
        for measurement in measurements:
            measurement += random.gauss(0,noise_vector)
    else:
        for measurement in measurements:
            for i,var in enumerate(noise_vector):
                measurement[i] += random.gauss(0,var)

    return measurements


def substitute_variables(model,variables,values):
    for variable,value in zip(variables,values):
        model = model.subs(variable,value)
    return model


def costfcn(measured_system,estimated_parameters,outputs,measured_values,parameters,output_variables):
    n = len(measurements)
    e = lambdify(measured_values+parameters+output_variables,(model - output_variables[0])**2)

    error = 0
    #for measurement,moutput in zip(measurements,output):
    #    error += e(*(measurement+list(x)),moutput)/n

    return np.float64(error)

#=============================================================#
# Model Tools                                                 #
#-------------------------------------------------------------#
def generateSimpleMichealisMentenModel(number_enzymes):
    return model,measurement_symbols,output_symbols,parameter_symbols

def generateRandomSystem(model,parameters,bounds=(0,10)):
    actual_system = model
    actual_parameters = []
    
    for parameter in parameters:
        #Generate a random parameter in bounds
        rnum = random.uniform(*bounds)
        actual_parameters.append(rnum)
        actual_system = actual_system.subs(parameter,rnum)        
    
    return actual_parameters,actual_system

#Generate a set of random measurements
def generateRandomMeasurements(actual_system,measurement_symbols,n):
    measurements = []
    outputs = []

    #Generate n Pathway Variant Measurements
    for i in range(n):
        measurement = []
        system = actual_system
        
        #Generate a random measurement
        for j,symbol in enumerate(measurement_symbols):
            measurement.append(random.uniform(0,100))
            system = system.subs(symbol,measurement[j])
        measurements.append(measurement)

        #Simulate System
        outputs.append(N(system))
    
    return measurements,outputs

#Generate a specific measurment
def generateMeasurement(actual_system,measurement,measurement_symbols):
    output = substitute_variables(actual_system,measurement_symbols,measurement)
    return output

def generateDiffEqMeasurement(f,t1,measurement,y0):
    g = lambda x,t: f(list(measurement)+list(x),t)
    output = odeint(g,y0,(0,t1))[-1][-1]
    return output

#Fit kinetic model with the known data
def fitKineticModel(model,measurements,measured_values,output_variables,output,parameters,actual_parameters,):    
    #Create Lambda Function from Existing Symbolic Manipulations
    n = len(measurements)
    e = lambdify(measured_values+parameters+output_variables,(model - output_variables[0])**2)
    
    #Vanilla Python Cost function
    def cost(x):
        error = 0
        #for measurement,moutput in zip(measurements,output):
        #    error += e(*(measurement+list(x)),moutput)/n
        return np.float64(error)
    
    #Run Optimization algorithm
    estimated_parameters = differential_evolution(cost,[(0,10),(0,10),(0,10),(0,10),(0,10)])
    #print(estimated_parameters.message)
    return estimated_parameters.x


def evaluate_model(model,measurements,msymbols,parameters,psymbols,outputs,osymbols):
    #Create a fast numerical model
    estimated_model = substitute_variables(model,psymbols,parameters)
    estimated_model = lambdify(msymbols,estimated_model)

    #Calculate model R^2 from test data
    mu = sum(outputs)/len(outputs)
    sst = 0
    ssr = 0
    for output,measurement in zip(outputs,measurements):
        sst += (output - mu)**2
        ssr += (output - estimated_model(*measurement))**2
    r_squared = 1 - ssr/sst
    
    return r_squared    


## Needed!
def findModelMax(model,bound,symbols):
    #determine if model is ml or kinetic
    if isinstance(model, mul.Mul):
        #Kinetic Model
        costfun = lambdify(symbols,-1*model)
        def cost(x):
            return costfun(*x)
        
        #print(lambdastr(symbols,-1*model))
        #print(cost([0.1,0.1]))
    else:
        #Machine Learning Model
        costfun = model.predict
        def cost(x):
            return -1*costfun(x)
    
    #Find Maximum using differential evolution
    bounds = []
    for _ in range(len(symbols)):
        bounds.append(bound)
        
    estimated_parameters = differential_evolution(cost,bounds,disp=False)
    
    return estimated_parameters.x, estimated_parameters.fun


#=============================================================#
# Visualization Tools                                         #
#-------------------------------------------------------------#

def modelPredictivePowerPlot(data,points):
    data = list(map(list, zip(*data)))
    n = len(data) + 1

    #Calculate Median of each data set
    medians = []
    for point in data:
        medians.append(statistics.median(point))

    #Calculate Median Expected Deviation
    meds = []
    for (point,median) in zip(data,medians):
        medlist = [abs(x - median) for x in point]
        meds.append(statistics.median(medlist))

    # example data
    x = points
    y = medians
    print('lengths:',len(x),len(y))
    print('data',x,y)
    # example variable error bar values
    yerr = meds
    #xerr = meds

    ## First illustrate basic pyplot interface, using defaults where possible.
    #plt.figure()
    #plt.errorbar(x, y, xerr=0.2, yerr=0.4)
    #plt.title("Simplest errorbars, 0.2 in x, 0.4 in y")

    # Now switch to a more OO interface to exercise more features.
    fig, axs = plt.subplots(nrows=1, ncols=1, sharex=True)
    ax = axs
    ax.errorbar(x, y, yerr=yerr)
    ax.set_title('Data Points Needed to Ensure Model Predictivity')
    ax.set_xlabel('Number of Data Points')
    ax.set_ylabel('R^2 of Model')

    plt.show()

def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower ofset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax/(vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highets point in the colormap's range.
          Defaults to 1.0 (no upper ofset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap

def plotModel(model,data,targets,midpoint=0.1,pcs=None,title=None,zlabel=None,ax=None):
    '''Plots a 2d projection of the model onto the principal components.
       The data is overlayed onto the model for visualization.
    '''
    
    #Visualize Model
    #Create Principal Compoenents for Visualiztion of High Dimentional Space
    pca = PCA(n_components=2)
    if pcs is not None:
        pca.components_ = pcs
        
    data_transformed = pca.fit_transform(data)
    
    
    #Get Data Range
    xmin = np.amin(data_transformed[:,0])
    xmax = np.amax(data_transformed[:,0])
    ymin = np.amin(data_transformed[:,1])
    ymax = np.amax(data_transformed[:,1])

    #Scale Plot Range
    scaling_factor = 0.5
    xmin = xmin - (xmax - xmin)*scaling_factor/2
    xmax = xmax + (xmax - xmin)*scaling_factor/2
    ymin = ymin - (ymax - ymin)*scaling_factor/2
    ymax = ymax + (ymax - ymin)*scaling_factor/2

    #Generate Points in transformed Space
    points = 1000
    x = np.linspace(xmin,xmax,num=points)
    y = np.linspace(ymin,ymax,num=points)
    xv, yv = np.meshgrid(x,y)

    #reshape data for inverse transform
    xyt = np.concatenate((xv.reshape([xv.size,1]),yv.reshape([yv.size,1])),axis=1)
    xy = pca.inverse_transform(xyt)
    
    #predict z values for plot
    z = model.predict(xy).reshape([points,points])
    minpoint = min([min(p) for p in z])
    maxpoint = max([max(p) for p in z])
    
    #Plot Contour from Model
    if ax is None:
        fig = plt.figure()
        ax = plt.gca()
    
    scaled_targets = [target/max(targets)*200 for target in targets]
    
    #Overlay Scatter Plot With Training Data
    ax.scatter(data_transformed[:,0],
                [1*value for value in data_transformed[:,1]],
                c='k',
                cmap=plt.cm.bwr,
                marker='+',
                s=scaled_targets,
                linewidths=1.5
                )
    
    ax.grid(b=False)

    midpercent = (midpoint-minpoint)/(maxpoint-minpoint)
    centered_cmap = shiftedColorMap(plt.cm.bwr, midpoint=midpercent)
    cmap = centered_cmap
    
    if midpercent > 1:
        midpercent = 1
        cmap = plt.cm.Blues_r
    elif midpercent < 0:
        midpercent = 0
        cmap = plt.cm.Reds
    
    z = [row for row in reversed(z)]
    im = ax.imshow(z,extent=[xmin,xmax,ymin,ymax],cmap=cmap,aspect='auto')
    
    if title is not None:
        ax.set_title(title)
    
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')

    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.05)
    if zlabel is not None:
        plt.colorbar(im, cax=cax,label=zlabel)
    else:
        plt.colorbar(im, cax=cax)
    

def plotPredictions(data,row_labels,column_labels,title=None):


     #Create Plot
     data = np.array(list(map(list, zip(*data))))
     cmap = plt.cm.bwr
     fig, ax = plt.subplots()
     heatmap = ax.pcolor(data, cmap=cmap)
     plt.colorbar(heatmap)

     for y in range(data.shape[0]):
         for x in range(data.shape[1]):
             plt.text(x + 0.5, y + 0.5, '%.4f' % data[y, x],
                      horizontalalignment='center',
                      verticalalignment='center',
                      )

     # put the major ticks at the middle of each cell
     ax.set_xticks(np.arange(data.shape[1])+0.5, minor=False)
     ax.set_yticks(np.arange(data.shape[0])+0.5, minor=False)

     # want a more natural, table-like display
     ax.invert_yaxis()
     #ax.xaxis.tick_top()

     ax.set_yticklabels(row_labels, minor=False)
     ax.set_xticklabels(column_labels, minor=False)
     if title is not None:
         plt.title(title)

     plt.show()


def plotBar(data,xlabels,ylabel,title,barlabels,r2=True):
    
    #Plot Expected Limonene Production from Candidate Test Points
    n_groups = len(data)
    index = np.arange(len(data[0]))

    bar_width = 0.35
    colors = ('b','r','g','y','k')
    error_config = {'ecolor': '0.3'}
    plt.figure(figsize=(8,5))
    opacity = 0.4

    #print(data)
    #print(index)
    for group in range(n_groups):
        #print('Group: ',group)
        plt.bar(1.5*index + group*bar_width + 0.5, data[group],bar_width,
                label=barlabels[group],
                alpha=opacity,
                color = colors[group])


    plt.xlabel('Model')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(1.5*index + bar_width + 0.5,xlabels)
    plt.legend()

    axes = plt.gca()
    if r2:
        axes.set_ylim(bottom=-0.01, top=1)

    plt.tight_layout()
    plt.show()

def generateReport(points,models,feature_labels=None,target_labels=None,bestPoints=4):
        
    model_labels = [model for model in models]
        
    #Calculate and Plot Prediction from Each Model
    values = []
    for point in points[:bestPoints]:
        model_values = []
        for model in models:
            model_values.append(models[model].predict(np.array(point).reshape(1, -1)))
        values.append(model_values)
        
    plotBar(values,model_labels,'','',model_labels,r2=False)
    
    #Print out conditions for each set of points.
    
    #Create Header
    if feature_labels is not None:
        header = 'No., '
        for label in feature_labels:
            header += label + ', '
        
        for label in model_labels:
            header += label + ', '
    
    lineformat = '{0}, '
    for i in range(len(points[0])+len(models)):
        lineformat += '{' + str(i+1) + ':.3f}, ' 
    
    print('<<<===== Best Points =====>>>')
    print(header)
    for i,point in enumerate(points):
        
        #Create Lines
        predictions = []
        for model in models:
            predictions.append(models[model].predict(np.array(point).reshape(1, -1)))
        line = [i+1,] + [float(item) for item in point] + [float(prediction) for prediction in predictions]
        
        if i == bestPoints:
            print('')
            print('<<<===== Test Points Which Reduce Model Variance =====>>>')
            print(header)
        
        #print(lineformat)
        #print(tuple(line))
        print(lineformat.format(*line))
        
    