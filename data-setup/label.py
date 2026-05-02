import trackpy
import pims
import matplotlib.pyplot as plt
import pandas as pd

def main():
    """Run a simple trackpy labeling check on a sample image."""
    # Load image (convert to grayscale if needed)
    try:
        frames = pims.open('your_image.jpg') 

        # trackpy.locate(image, diameter, minmass, separation)
        # diameter: estimate of particle size in pixels (looks like ~15-19px here)
        # invert: Set to True if particles are darker than background (here they are bright)
        features = trackpy.locate(frames[0], diameter=19, invert=False, minmass=200)

        # Visualize to check accuracy
        plt.figure()
        trackpy.annotate(features, frames[0])
        plt.show()
    except FileNotFoundError:
        print("Error: 'your_image.jpg' not found. Please provide a valid image path.")

if __name__ == "__main__":
    main()
