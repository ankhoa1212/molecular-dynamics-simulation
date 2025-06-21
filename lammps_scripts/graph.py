# Create scatter plot of nodes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import mdtraj as md
import os

def generate_csv(filename, csv_filename):
    print(f"Opening file {filename}...")
    with open(filename, 'r') as f:
        lines = f.readlines()
        print(f"Reading data from {filename}...")
        read_data, read_time = False, False
        values = None
        timestep = None
        for line in lines:
            if line.startswith('ITEM:'):
                if line.startswith('ITEM: TIMESTEP'):
                    read_time = True
                    read_data = False
                elif line.startswith('ITEM: ATOMS'):
                    if not values:
                        values = line[12:].split()
                        values.append('timestep')
                        data = pd.DataFrame(columns=values)
                    read_data = True
                continue
            if read_time:
                timestep = line
                read_time = False
                print(f"timestep: {timestep}")
                continue
            elif read_data:
                input_line = line.split()
                input_line.append(timestep)
                temp = [float(x) if i > 1 and i < 8 else int(x) for i,x in enumerate(input_line)]
                print(f"Adding data at timestep {timestep}: {len(temp)} {temp}")
                data.loc[len(data)] = temp

        # Save the dataframe to a CSV file
        data.to_csv(csv_filename, index=False)
        print(f"Data saved to {csv_filename}")

# Read data from a raw file
if len(sys.argv) < 2:
    print("Usage: python graph.py <filename>")
    sys.exit(1)

filename = sys.argv[1]

data = pd.DataFrame()

csv_file = filename.rsplit('.', 1)[0] + '.csv'
if os.path.exists(csv_file):
    print(f"CSV file {csv_file} already exists.")
    print("Reading in csv data...")
    data = pd.read_csv(csv_file)
else:
    generate_csv(filename, csv_file)

print("Creating plots...")
if 'vx' in data.columns and 'vy' in data.columns and 'timestep' in data.columns:
    data['v_mag'] = np.sqrt(data['vx']**2 + data['vy']**2)
    avg_vmag = data.groupby('timestep')['v_mag'].mean()
    plt.figure()
    plt.plot(avg_vmag.index, avg_vmag.values, marker='o')
    plt.xlabel('Timestep')
    plt.ylabel('Average Velocity Magnitude')
    plt.title('Average Velocity Magnitude per Timestep')
    plt.savefig(filename.rsplit('.', 1)[0] + '_avg_velocity_magnitude.png')
    plt.show()
else:
    print("Required columns ('vx', 'vy', 'timestep') not found in data.")
