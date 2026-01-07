import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from phase_diagram import extract_epsilon_and_molecules

def plot_velocity_over_time(filename, output_dir):
    timesteps = []
    avg_velocities = []
    
    current_velocities = []
    reading_atoms = False

    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('ITEM: TIMESTEP'):
                # If we just finished a frame, calculate the average
                if current_velocities:
                    avg_velocities.append(np.mean(current_velocities))
                    current_velocities = []
                reading_atoms = False
                timesteps.append(int(next(f)))
            
            elif line.startswith('ITEM: ATOMS'):
                reading_atoms = True
                # Identify column indices for vx and vy
                header = line.split()[2:]
                vx_idx = header.index('vx')
                vy_idx = header.index('vy')
                continue
            
            elif reading_atoms:
                parts = line.split()
                if not parts: continue
                vx = float(parts[vx_idx])
                vy = float(parts[vy_idx])
                # Calculate magnitude: sqrt(vx^2 + vy^2)
                v_mag = np.sqrt(vx**2 + vy**2)
                current_velocities.append(v_mag)

        # Append the final frame
        if current_velocities:
            avg_velocities.append(np.mean(current_velocities))

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, avg_velocities, marker='o', linestyle='-', color='b')
    plt.xlabel('Timestep')
    plt.ylabel('Mean Velocity Magnitude')
    plt.title('Average Velocity Magnitude Over Time')
    plt.grid(True, linestyle='--', alpha=0.7)
    molecules, epsilon = extract_epsilon_and_molecules(filename)
    
    i = filename.find('.')
    if i != -1:
        filename = filename[:i]
    filename = f"{filename}_{molecules}_{epsilon}_velocity_graph.png"
    output = filename
    if output_dir:
        output = os.path.join(output_dir, filename)
    plt.savefig(output)
    print(f"Graph saved to {output}")
    plt.show()

# Replace 'your_file.lammpstrj' with your actual filename
# plot_velocity_over_time('your_file.lammpstrj')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot velocity over time from LAMMPS trajectory file')
    parser.add_argument('--filename', '-f', default='/home/austin/git/molecular-dynamics-simulation/lammps-scripts/test_same/test.in_100_5.0.lammpstrj', help='Path to the LAMMPS trajectory file')
    parser.add_argument('--output_dir', default=None, help='output directory to save graph file to')
    args = parser.parse_args()
    
    plot_velocity_over_time(args.filename, args.output_dir)