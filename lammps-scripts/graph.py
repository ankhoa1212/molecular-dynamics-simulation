"""
graph.py
This module processes molecular dynamics simulation output files (LAMMPS dump format),
extracts atomic data, and generates a CSV file for further analysis. It also provides
visualization of the average velocity magnitude per timestep using matplotlib.
Functions:
    check_is_float(value):
        Checks if the given value can be converted to a float.
    check_is_int(value):
        Checks if the given value can be converted to an integer.
    generate_csv(filename, csv_filename):
        Reads a LAMMPS dump file, extracts atomic data for each timestep,
        and saves it as a CSV file with appropriate columns.
Script Usage:
    python graph.py <filename>
        - If a corresponding CSV file does not exist, it will be generated from the raw file.
        - If the CSV exists, it will be loaded for analysis.
        - The script computes and plots the average velocity magnitude ('vx', 'vy') per timestep.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import mdtraj as md


def check_is_float(value):
    """ Check if a value is a float """
    try:
        float(value)
        return True
    except ValueError:
        return False

def check_is_int(value):
    """ Check if a value is an integer """
    try:
        int(value)
        return True
    except ValueError:
        return False

def generate_csv(filename, csv_filename):
    """ Generate a CSV file from a LAMMPS dump file """
    print(f"Opening file {filename}...")
    read_data, read_time = False, False
    data = pd.DataFrame()
    with open(filename, "r", encoding="utf-8") as file:
        lines = file.readlines()
        print(f"Reading data from {filename}...")
        values = None
        timestep = None
        for line in lines:
            if line.startswith("ITEM:"):
                if line.startswith("ITEM: TIMESTEP"):
                    read_time = True
                    read_data = False
                elif line.startswith("ITEM: ATOMS"):
                    # set values to column names, skip the first 12 characters "ITEM: ATOMS ""
                    if not values:
                        values = line[12:].split()
                        values.append("timestep")
                        data = pd.DataFrame(columns=values)
                        print(values)
                    read_data = True
                continue
            if read_time:
                timestep = line
                read_time = False
                print(f"timestep: {timestep}")
            elif read_data:
                input_line = line.split()
                input_line.append(timestep)
                temp_data = []
                for value in input_line:
                    if check_is_float(value):
                        temp_data.append(float(value))
                    elif check_is_int(value):
                        temp_data.append(int(value))
                    else:
                        temp_data.append(value)
                print(
                    f"Adding data at timestep {timestep}: {len(temp_data)} {temp_data}"
                )
                data.loc[len(data)] = temp_data

        # Save the dataframe to a CSV file
        data.to_csv(csv_filename, index=False)
        print(f"Data saved to {csv_filename}")
        return data

def read_data_from_file():
    """ Read data from a file or generate a CSV if it doesn't exist """
    # Read data from a raw file
    if len(sys.argv) < 2:
        print("Usage: python graph.py <filename>")
        sys.exit(1)

    filename = sys.argv[1]

    data = pd.DataFrame()

    csv_file = filename.rsplit(".", 1)[0] + ".csv"
    if os.path.exists(csv_file):
        print(f"CSV file {csv_file} found.")
        print("Reading in csv data...")
        data = pd.read_csv(csv_file)
    else:
        data = generate_csv(filename, csv_file)
    return data, filename

def create_plots(data, filename):
    """ Create plots from the data """
    print("Creating plots...")
    if "vx" in data.columns and "vy" in data.columns and "timestep" in data.columns:
        data["v_mag"] = np.sqrt(data["vx"] ** 2 + data["vy"] ** 2)
        avg_vmag = data.groupby("timestep")["v_mag"].mean()
        plt.figure()
        plt.plot(avg_vmag.index, avg_vmag.values, marker="o")
        plt.xlabel("Timestep")
        plt.ylabel("Average Velocity Magnitude")
        plt.title("Average Velocity Magnitude per Timestep")
        plt.savefig(filename.rsplit(".", 1)[0] + "_avg_velocity_magnitude.png")
        plt.show()
    else:
        print("Required columns ('vx', 'vy', 'timestep') not found in data.")

def main():
    """ Main function to execute the script """
    data, filename = read_data_from_file()
    create_plots(data, filename)

if __name__ == "__main__":
    main()
