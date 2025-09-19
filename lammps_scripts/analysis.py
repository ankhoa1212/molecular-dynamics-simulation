# pylint: skip-file
import MDAnalysis as mda
import MDAnalysis.analysis.msd as msd
import numpy as np
import matplotlib.pyplot as plt

# Load a LAMMPS dump file
universe = mda.Universe("results/central_pair_interaction.in.lammpstrj", topology_format='LAMMPSDUMP')

# Select the atoms for which you want to calculate the MSD
# Example: Select all atoms of a specific residue, e.g., 'SOL' for solvent
selected_atoms = universe.select_atoms("type 1")

# Set up the EinsteinMSD analysis
# 'msd_type' can be 'xyz', 'xy', 'x', etc.
# 'fft=True' uses a faster algorithm, which requires `tidynamics`
MSD_analysis = msd.EinsteinMSD(selected_atoms, msd_type='xyz', fft=True)

# Run the analysis over the trajectory
MSD_analysis.run()

# Access the results
# The .timeseries attribute holds the average MSD over all selected particles
msd_data = MSD_analysis.results.timeseries
# The .msds_by_particle attribute holds the individual MSD for each particle
individual_msd = MSD_analysis.results.msds_by_particle

# Create the lag-time axis for plotting
n_frames = MSD_analysis.n_frames
# Get the timestep from the universe, or set it manually if it's not present
if universe.trajectory.dt is not None:
    timestep = universe.trajectory.dt
else:
    timestep = 1.0  # Or your timestep in ps
lagtimes = np.arange(n_frames) * timestep

# Plotting the MSD
fig, ax = plt.subplots()
ax.plot(lagtimes, msd_data, label=f'MSD for {selected_atoms.n_atoms} particles')
ax.set_xlabel('Lag Time (ps)')
ax.set_ylabel(r'MSD ($\AA^2$)')
ax.set_title('Mean Squared Displacement')
ax.legend()
plt.show()

# You can also compute the diffusion coefficient (D) from the linear part of the MSD plot
# D = (slope / 6) for a 3D MSD
from scipy.stats import linregress
# Find the linear region of your MSD data and fit it
# For example, use a range from index 10 to 50
start_index = 10
end_index = 50
linear_model = linregress(lagtimes[start_index:end_index], msd_data[start_index:end_index])
slope = linear_model.slope
diffusion_coefficient = slope / (2 * MSD_analysis.dim_fac)
print(f"Diffusion Coefficient: {diffusion_coefficient} Å²/ps")

def calculate_avg_velocity(universe, atomgroup):
    """
    Calculates the average velocity of an AtomGroup over a trajectory.
    """
    velocities = []

    # Get the position of the first frame
    universe.trajectory[0]
    prev_positions = atomgroup.positions

    # Iterate through the remaining frames
    for ts in universe.trajectory[1:]:
        dt = ts.dt
        current_positions = atomgroup.positions

        # Calculate displacement and instantaneous velocity
        displacement = current_positions - prev_positions
        instantaneous_velocity = displacement / dt

        # Store the velocity for the current time step
        velocities.append(np.mean(instantaneous_velocity, axis=0))

        # Update positions for the next iteration
        prev_positions = current_positions

    # Plot the average velocity over time.
    plt.figure(figsize=(10, 6))
    plt.plot([x for x in range(len(velocities))], [[x, y] for x, y, z in velocities]) # plot x and y velocities
    
    # Add labels and title.
    plt.title(f'Average Velocity Magnitude over Time')
    plt.xlabel('Steps')
    plt.ylabel('Average Velocity')
    plt.grid(True)

    # Display the plot.
    plt.show()

# Example usage:
# Create a Universe object from your topology and trajectory files
# u = mda.Universe("topology.pdb", "trajectory.xtc")

# Select the atoms for analysis (e.g., all atoms)
# protein = u.select_atoms("protein")

# Calculate the average velocity
avg_vel = calculate_avg_velocity(universe, selected_atoms)
print(f"Average velocity vector: {avg_vel} Å/ps")
print(f"Average velocity magnitude: {np.linalg.norm(avg_vel)} Å/ps")

if __name__ == "__main__":
    avg_vel = calculate_avg_velocity(universe, selected_atoms)
    print(f"Average velocity vector: {avg_vel} Å/ps")
    print(f"Average velocity magnitude: {np.linalg.norm(avg_vel)} Å/ps")