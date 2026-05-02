#!/usr/bin/env python
# coding: utf-8

# In[2]:


import ast
import numpy as np
import pandas as pd
from pandas import DataFrame, Series  # for convenience
#import trackpy
from pandas import DataFrame, Series  # for convenience
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.neighbors import NearestNeighbors
import pims
import math
from fractions import Fraction
from array import array
import matplotlib.patches as mpatches 
import imageio
import os


# In[3]:


def getPoints(txtfile):
    times = np.loadtxt(txtfile, skiprows = 1)
    return times[:,1:3]


# Draw the voronoi diagram thingy

# In[4]:


def drawVoronoi(points2):
    vor = Voronoi(points2) 

    fig = voronoi_plot_2d(vor,point_size = 1.5, line_width = 0.5, line_colors = "black", show_vertices = False)
    #Change these parameters based on the size of the image
    """
    plt.ylim(400,0)
    plt.xlim(0,600)
    """
    #fig.set_size_inches(5,5)
    fig.set_dpi(300)
    for region in vor.regions:
        if not -1 in region:
            polygon = [vor.vertices[i] for i in region]
            if len(region) == 4:
                #Fills in Indigo
                plt.fill(*zip(*polygon),facecolor='#4B0082')
            elif len(region) == 5:
                #Fills in Cyan
                plt.fill(*zip(*polygon),facecolor='#00FFFF')
            elif len(region) == 6: 
                #Fills in Blue
                plt.fill(*zip(*polygon),facecolor='#0000FF')
            elif len(region) == 7:
                #Fills in Chartreuse
                plt.fill(*zip(*polygon),facecolor="#C1F80A")
            elif len(region) == 8:
            #Fills in Fuchsia
                plt.fill(*zip(*polygon),facecolor='#ED0DD9')
            elif len(region) == 9:
            #Fills in Grey
                plt.fill(*zip(*polygon),facecolor='#808080')
            else:
                #Fills in Green
                plt.fill(*zip(*polygon),facecolor='#15B01A')

    plt.title('Voronoi Diagram')

    #Colors and labels for each polygon
    colors = ['#4B0082', '#00FFFF', '#0000FF', '#C1F80A', '#ED0DD9', '#808080', '#15B01A']
    labels = ['4-sided polygon', '5-sided polygon', '6-sided polygon', '7-sided polygon', '8-sided polygon', '9-sided polygon', '10+ sided polygon']

    #Legend with the custom colors and labels
    patches = [mpatches.Patch(color=color, label=label) for color, label in zip(colors, labels)]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.show()


# histogram

# In[7]:


def hexaticOrder(points2):
    nbrs = NearestNeighbors(n_neighbors=7, algorithm='ball_tree').fit(points2)
    distances, indices = nbrs.kneighbors(points2)

    #Empty Angle_array list is initialized to store angle values for each point and its neighbors
    Angle_array = []
    #Loops through each point in points2 using range function
    for i in range(len(indices)):
        #For each point code initializes empty angle_array_row to store angles between that point and its neighbors.
        angle_array_row = []
        #loops through seven nearest neighbors of the current point
        for j in range(7):
            #For each neighbor, code retrieves the index of the neighbor from the indices array
            index_neighbor =  indices[i][j]
            #Extracts x and y coordinates of both the current point and its neighbor from the points2 array
            neighbor_x = points2[index_neighbor,0]
            neighbor_y = points2[index_neighbor,1]
            original_x = points2[i,0]
            original_y = points2[i,1]
            #computes difference for x and y coordinates between the point and its neighbor into delta_x and delta_y variables
            delta_x = neighbor_x - original_x
            delta_y = neighbor_y - original_y
            #if delta_x is not equal to 0, compute angle between point and its neighbor then append result to angle_array_row list
            #if delta_x is equal to 0, the code skips the calculation
            if delta_x != 0:
                #computes angle (radians) between the line connecting two points (original_x, original_y) and (neighbor_x, neighbor_y)
                angle = math.atan(delta_y/delta_x)
                #Once 7 neigh
                # bors of the point have been processed, the angle_array_row list is appended to the Angle_array list
                angle_array_row.append(angle)
        Angle_array.append(angle_array_row)
    #loop continues for all points in the points2 array.

    #Empty List to store hexactic order parameters for each row of angles
    hexatic_order_params_exp1_14 = []
    #Code loops through each row of angles in Angle_Array
    #print(Angle_array)
    for i in range(len(Angle_array)):
        #for each row, it determines number of angles in that row   
        #initializes hex_sum to store the sum of the exponential factors that define the hexatic order parameter
        hex_sum = 0
        for j in range(len(Angle_array[i])):
            #Calculates the exponential factor for that angle, note j is imaginary number
            hex_sum += np.exp(complex(0,6*Angle_array[i][j]))

        #After angles in row are processed, calculate hex_sum magnitude divided by 6   
        hexatic_order_params_exp1_14.append(abs(hex_sum)/6)
    return hexatic_order_params_exp1_14


# In[8]:


def drawHistogram(points2):    

    hexatic_order_params_exp1_14=hexaticOrder(points2)
    #Generate histogram
    plt.hist(hexatic_order_params_exp1_14, bins=10, color='magenta', alpha=0.9)

    # Set plot title and axes labels
    plt.title('Hexatic Order Parameters')
    plt.xlabel('Hexatic Order Parameter')
    plt.ylabel('Frequency')
    # Display Histogram
    plt.show()


# In[9]:


def printHexInfo(points2):
    hexatic_order_params_exp1_14=hexaticOrder(points2)
    hist, edges = np.histogram(hexatic_order_params_exp1_14, bins=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    for i in range(len(hist)):
        bin_label = "Hexatic Order: {:.1f} - {:.1f}; Quantity: ".format(edges[i], edges[i+1])
        print(bin_label + str(hist[i]))

    try:
        with open('hexatic_order.txt', 'r') as f:
            hexatic_order_params_exp1_14 = ast.literal_eval(f.read())
        # Calculate the mean of the variable
        mean = np.mean(hexatic_order_params_exp1_14)
    except FileNotFoundError:
        print("File not found. Please make sure the file exists and try again.")


# get files

# In[10]:


def getFiles(dir):
    files=os.listdir(dir)
    def sortalg(a):
        try:
            if("_" in a):
                return int(a[:-6])
            return int(a[:-4])
        except:
            return 0
    files.sort(key=sortalg)
    return files


# make the line graph

# In[12]:


def plotList(list):
    markers=['o','*','.','x','X','+','P','s','D','d','p','H','h','v','^','<','>','1','2','3','4','|','_']
    mark=0
    for f in list:
        if not os.path.isdir(f): #folder of our txt files
            continue
        x_axis=[]
        y_axis=[]
        files=getFiles(f)
        for file in files:
            if(file[-3:]=="txt"): #if is text file
                try:
                    points=getPoints(f+"/"+file)
                    hex=hexaticOrder(points) #sometimes this code throws an exception -> ignore it for now
                    y=np.mean(hex)
                    y_axis.append(y)
                    if("_" in file): #sometimes file will be ###_#.txt instead of just ##.txt
                        x=int(file[:-6])
                    else:
                        x=int(file[:-4])
                    #print(x)
                    x_axis.append(x)  
                except:
                    pass #😌
        plt.plot(x_axis,y_axis,marker=markers[mark%len(markers)],label=f)
        mark+=1
    plt.xlabel("Time (seconds)")
    plt.ylabel("mean hexatic order")
    plt.title("Hexatic order over time")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid()
    plt.show()


# In[13]:


def plotSingle(dir):
    plotList([dir])


# In[14]:


def plotAll():
    plotList(os.listdir())


# In[17]:


plotAll()
#plotSingle("70%")
#for fun
points=getPoints("100%/600.txt")
#print(points)
drawVoronoi(points)
drawHistogram(points)


# In[16]:


files=os.listdir()
for f in files:
    if(os.path.isdir(f)):
        plotSingle(f)


# In[27]:


diagrams=os.listdir()

for d in diagrams:
    if(not os.path.isdir(d)): continue
    files=os.listdir(d)
    for f in files:
        print(d+"/"+f)
        try:        
            points=getPoints(d+"/"+f)
            #print(points)
            drawVoronoi(points)
        except Exception as e:
            print(e)
            pass
        #drawHistogram(points)

