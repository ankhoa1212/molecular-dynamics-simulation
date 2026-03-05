"""Script to generate a scatter plot of Molecule Count vs. Epsilon
Requires Pandas and Matplotlib libraries.
To install the required libraries, run:
pip install pandas matplotlib
"""

import os

import matplotlib.colors
import matplotlib.pyplot as plt
import pandas as pd

# 1. Load the data from a CSV file
# Make sure CSV file has columns named exactly:
# 'molecule count', 'epsilon', and 'result'
df = pd.read_csv(os.path.join(os.getcwd(), "simulation_results.csv"))

# 2. Define a custom color map: Red for 0, Yellow for 1, Green for 2
colors = ["red", "yellow", "green"]
# Create a Colormap object from the list of colors
cmap = matplotlib.colors.ListedColormap(colors)
# Define the bounds for the colors (0 to 1, 1 to 2, 2 to 3)
bounds = [0, 1, 2, 3]
# Normalize the color mapping to the defined bounds
norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

# 3. Create the scatter plot
plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    df["epsilon"],  # X-axis data
    df["molecule count"],  # Y-axis data
    c=df["result"],  # Color points based on the 'result' column
    cmap=cmap,  # Use the custom color map
    norm=norm,  # Use the custom normalization
    s=50,  # Point size
    alpha=0.8,  # Transparency
)

# 4. Add labels and title
plt.xlabel(r"$\epsilon$ (Epsilon)", fontsize=14)
plt.ylabel("Molecule Count", fontsize=14)
plt.title("Molecule Count vs. $\\epsilon$ (Epsilon) (Result in Color)", fontsize=12)

# 5. Add a color bar
cbar = plt.colorbar(scatter, ticks=[0.5, 1.5, 2.5])
cbar.set_label("Result (0=Red, 1=Yellow, 2=Green)")
# Set the tick labels explicitly to match the results
cbar.set_ticklabels(
    ["0 (Not crystallized)", "1 (Partially crystallized)", "2 (Fully crystallized)"]
)

# 6. Save the plot as a PNG file
plt.savefig("molecule_count_vs_epsilon.png", dpi=300, bbox_inches="tight")

# 7. Display the plot
plt.grid(True)
plt.show()
