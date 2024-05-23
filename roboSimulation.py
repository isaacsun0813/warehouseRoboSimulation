import numpy as np
import matplotlib.pyplot as plt
import simpy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout

# Define parameters
num_robots = 50  # Increase the number of robots to generate more data
mass = 5  # mass of the object in kg
height = 2  # height to lift in meters
distance = 10  # distance to move horizontally in meters
g = 9.81  # acceleration due to gravity in m/s^2
power_rating = 100  # power rating of the motor in watts
battery_capacity = 1000  # battery capacity in watt-hours
initial_battery_level = 1000  # initial battery level in watt-hours

# Define material properties
materials = [
    {'type': 'aluminum', 'density': 2.7, 'efficiency': 0.9},
    {'type': 'steel', 'density': 7.8, 'efficiency': 0.8},
    {'type': 'plastic', 'density': 0.9, 'efficiency': 0.95}
]

# Simulate the environment using simpy
def robot(env, robot_id, material, power_data, battery_data, mass, height, distance, g, power_rating):
    while battery_data[robot_id][-1] > 0:
        task_type = np.random.choice(['lift', 'move'])
        if task_type == 'lift':
            work_done = mass * g * height
            time_to_complete = np.random.uniform(3, 7)  # random lift time between 3 and 7 seconds
        else:
            work_done = mass * distance
            time_to_complete = np.random.uniform(5, 15)  # random move time between 5 and 15 seconds

        power_consumed = min(work_done / time_to_complete / material['efficiency'], power_rating)
        battery_level = battery_data[robot_id][-1] - power_consumed * time_to_complete / 3600
        if battery_level < 0:
            battery_level = 0
        battery_data[robot_id].append(battery_level)
        power_data.append((env.now, robot_id, power_consumed, task_type, material['type'], battery_level))
        yield env.timeout(time_to_complete)

# Function to run multiple simulations
def run_simulations(num_simulations, num_robots, materials):
    all_power_data = []
    all_battery_data = []
    for _ in range(num_simulations):
        env = simpy.Environment()
        power_data = []
        battery_data = [[initial_battery_level] for _ in range(num_robots)]
        for i in range(num_robots):
            material = np.random.choice(materials)
            env.process(robot(env, i, material, power_data, battery_data, mass, height, distance, g, power_rating))
        env.run(until=300)  # run the simulation for 300 seconds
        all_power_data.extend(power_data)
        all_battery_data.extend(battery_data)
    return all_power_data, all_battery_data

# Run the simulations
num_simulations = 10  # Increase the number of simulations
all_power_data, all_battery_data = run_simulations(num_simulations, num_robots, materials)

# Prepare data for machine learning
times = np.array([data[0] for data in all_power_data]).reshape(-1, 1)
robot_ids = np.array([data[1] for data in all_power_data]).reshape(-1, 1)
powers = np.array([data[2] for data in all_power_data])
task_types = np.array([1 if data[3] == 'lift' else 0 for data in all_power_data]).reshape(-1, 1)
material_types = np.array([data[4] for data in all_power_data])
material_densities = np.array([next(item['density'] for item in materials if item['type'] == mt) for mt in material_types]).reshape(-1, 1)
material_efficiencies = np.array([next(item['efficiency'] for item in materials if item['type'] == mt) for mt in material_types]).reshape(-1, 1)
battery_levels = np.array([data[5] for data in all_power_data]).reshape(-1, 1)

# Combine features and split data
X = np.hstack((times, robot_ids, task_types, material_densities, material_efficiencies, battery_levels))
y = powers
X_train, X_test, y_train, y_test, material_types_train, material_types_test = train_test_split(X, y, material_types, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build and train a neural network model
model = Sequential([
    Input(shape=(6,)),  # Use Input layer with the shape of the input data
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train_scaled, y_train, epochs=100, batch_size=10, verbose=1, validation_split=0.2)

# Make predictions
pred_powers = model.predict(X_test_scaled)

# Plot the results
for material in materials:
    material_type = material['type']
    material_mask = material_types_test == material_type

    plt.figure(figsize=(10, 6))
    plt.scatter(X_test[material_mask, 0], y_test[material_mask], label=f'Observed Power Consumption ({material_type})', alpha=0.3)
    plt.scatter(X_test[material_mask, 0], pred_powers[material_mask], label=f'Predicted Power Consumption ({material_type})', alpha=0.3)

    plt.xlabel('Time (s)')
    plt.ylabel('Power (W)')
    plt.title(f'Power Consumption during Various Tasks by Robots with {material_type.capitalize()}')
    plt.legend()
    plt.show()

# Efficiency prediction for a specific scenario
new_task_time = 10  # time for a new task in seconds
new_robot_id = 1  # id for the new robot
new_task_type = 1  # task type (1 for lift, 0 for move)
new_material_density = 2.7  # material density for the new task
new_material_efficiency = 0.9  # material efficiency for the new task
new_battery_level = 950  # battery level for a new task in watt-hours

# Convert the input to a numpy array and scale it
new_task_input = scaler.transform(np.array([[new_task_time, new_robot_id, new_task_type, new_material_density, new_material_efficiency, new_battery_level]]))

predicted_power = model.predict(new_task_input)
print(f"Predicted Power Consumption for a {new_task_time}-second task by robot {new_robot_id} with material density {new_material_density} and efficiency {new_material_efficiency}: {predicted_power[0][0]:.2f} Watts")

