import numpy as np
import matplotlib.pyplot as plt

# Define parameters
num_robots = 5  # number of robots
mass = 5  # mass of the object in kg
height = 2  # height to lift in meters
distance = 10  # distance to move horizontally in meters
g = 9.81  # acceleration due to gravity in m/s^2
power_rating = 100  # power rating of the motor in watts
battery_capacity = 1000  # battery capacity in watt-hours
initial_battery_level = 1000  # initial battery level in watt-hours
carbon_intensity = 0.4  # kg CO2 per kWh

# Define material properties with more accurate data
materials = [
    {'type': 'aluminum', 'density': 2.70, 'efficiency': 0.92, 'embodied_carbon': 8.24},  # kg CO2 per kg
    {'type': 'steel', 'density': 7.85, 'efficiency': 0.85, 'embodied_carbon': 2.34},  # kg CO2 per kg
    {'type': 'plastic', 'density': 0.95, 'efficiency': 0.90, 'embodied_carbon': 6.15}  # kg CO2 per kg
]

# Plot the number of robots
plt.figure(figsize=(8, 6))
plt.bar(['Number of Robots'], [num_robots], color='blue', width=0.5)
plt.ylabel('Count')
plt.title('Number of Robots')
plt.show()

# Plot the physical parameters of the tasks
plt.figure(figsize=(14, 8))
physical_params = ['Mass (kg)', 'Height (m)', 'Distance (m)', 'Gravity (m/s^2)', 'Power Rating (W)', 'Battery Capacity (Wh)', 'Initial Battery Level (Wh)', 'Carbon Intensity (kg CO2/kWh)']
physical_values = [mass, height, distance, g, power_rating, battery_capacity, initial_battery_level, carbon_intensity]
plt.bar(physical_params, physical_values, color='green', width=0.5)
plt.ylabel('Values', fontsize=10)
plt.title('Physical Parameters of the Tasks', fontsize=12)
plt.xticks(fontsize=6)
plt.yticks(fontsize=8)
plt.show()


# Plot the material properties
plt.figure(figsize=(12, 8))
material_types = [material['type'] for material in materials]
densities = [material['density'] for material in materials]
efficiencies = [material['efficiency'] for material in materials]
embodied_carbons = [material['embodied_carbon'] for material in materials]

x = np.arange(len(material_types))

width = 0.2  # the width of the bars
plt.bar(x - width, densities, width, label='Density (kg/m³)', color='purple')
plt.bar(x, efficiencies, width, label='Efficiency', color='black')
plt.bar(x + width, embodied_carbons, width, label='Embodied Carbon (kg CO2/kg)', color='grey')

plt.xlabel('Material Type')
plt.ylabel('Values')
plt.title('Material Properties')
plt.xticks(x, material_types)
plt.legend()
plt.show()
