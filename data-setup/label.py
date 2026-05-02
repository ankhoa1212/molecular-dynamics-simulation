import trackpy
import pims
import matplotlib.pyplot as plt
import pandas as pd

# Load image (convert to grayscale if needed)
frames = pims.open('your_image.jpg') 

# trackpy.locate(image, diameter, minmass, separation)
# diameter: estimate of particle size in pixels (looks like ~15-19px here)
# invert: Set to True if particles are darker than background (here they are bright)
f = trackpy.locate(frames[0], diameter=19, invert=False, minmass=200)

# Visualize to check accuracy
plt.figure()
trackpy.annotate(f, frames[0])