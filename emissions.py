import numpy as np
import matplotlib.pyplot as plt
import simpy
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

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

# Simulate the environment using simpy
def robot(env, robot_id, material, power_data, carbon_data, battery_data, mass, height, distance, g, power_rating, carbon_intensity):
    embodied_carbon = material['density'] * mass * material['embodied_carbon']
    while battery_data[robot_id][-1] > 0:
        task_type = np.random.choice(['lift', 'move'])
        if task_type == 'lift':
            work_done = mass * g * height
            time_to_complete = np.random.uniform(3, 10)  # increased variation in lift time
        else:
            work_done = mass * distance
            time_to_complete = np.random.uniform(5, 20)  # increased variation in move time

        power_consumed = min(work_done / time_to_complete / material['efficiency'], power_rating)
        operational_carbon = power_consumed * time_to_complete / 3600 * carbon_intensity
        total_carbon = embodied_carbon + operational_carbon

        battery_data[robot_id].append(battery_data[robot_id][-1] - power_consumed * time_to_complete / 3600)
        power_data.append((env.now, robot_id, power_consumed, task_type, material['type'], battery_data[robot_id][-1]))
        carbon_data.append((env.now, robot_id, total_carbon, task_type, material['type']))
        yield env.timeout(time_to_complete)

# Set up the simulation environment
env = simpy.Environment()
power_data = []
carbon_data = []
battery_data = [[initial_battery_level] for _ in range(num_robots)]
for i in range(num_robots):
    material = np.random.choice(materials)
    env.process(robot(env, i, material, power_data, carbon_data, battery_data, mass, height, distance, g, power_rating, carbon_intensity))
env.run(until=1000)  # run the simulation for a longer duration

# Prepare data for machine learning
times = np.array([data[0] for data in power_data]).reshape(-1, 1)
robot_ids = np.array([data[1] for data in power_data]).reshape(-1, 1)
powers = np.array([data[2] for data in power_data])
task_types = np.array([1 if data[3] == 'lift' else 0 for data in power_data]).reshape(-1, 1)
material_types = np.array([data[4] for data in power_data])
material_densities = np.array([next(item['density'] for item in materials if item['type'] == mt) for mt in material_types]).reshape(-1, 1)
material_efficiencies = np.array([next(item['efficiency'] for item in materials if item['type'] == mt) for mt in material_types]).reshape(-1, 1)
material_embodied_carbon = np.array([next(item['embodied_carbon'] for item in materials if item['type'] == mt) for mt in material_types]).reshape(-1, 1)
battery_levels = np.array([data[5] for data in power_data]).reshape(-1, 1)
total_carbons = np.array([data[2] for data in carbon_data])

# Combine features and split data
X = np.hstack((times, robot_ids, task_types, material_densities, material_efficiencies, material_embodied_carbon, battery_levels))
y = total_carbons
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train a neural network model
model = Sequential()
model.add(Input(shape=(7,)))  # Define the input shape
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=1, validation_split=0.2)

# Make predictions
pred_carbons = model.predict(X_test)

# Separate data by material and task type
def filter_data(task_type, material_type):
    obs_data = [(X_test[i, 0], y_test[i]) for i in range(len(y_test)) if task_types[i] == task_type and material_types[i] == material_type]
    pred_data = [(X_test[i, 0], pred_carbons[i]) for i in range(len(pred_carbons)) if task_types[i] == task_type and material_types[i] == material_type]
    return obs_data, pred_data

obs_lift_plastic, pred_lift_plastic = filter_data(1, 'plastic')
obs_move_plastic, pred_move_plastic = filter_data(0, 'plastic')
obs_lift_aluminum, pred_lift_aluminum = filter_data(1, 'aluminum')
obs_move_aluminum, pred_move_aluminum = filter_data(0, 'aluminum')
obs_lift_steel, pred_lift_steel = filter_data(1, 'steel')
obs_move_steel, pred_move_steel = filter_data(0, 'steel')

# Calculate cumulative carbon consumption
def accumulate(data, interval):
    if not data:
        return np.array([]), np.array([])
    data_sorted = sorted(data)
    times, carbons = zip(*data_sorted) if data_sorted else ([], [])
    cumulative_carbons = np.cumsum(carbons)
    time_bins = np.arange(0, max(times) + interval, interval)
    cumulative_bins = np.interp(time_bins, times, cumulative_carbons)
    return time_bins, cumulative_bins

interval = 30  # define the interval for the graph

# Accumulate data for plotting
obs_lift_plastic_bins, obs_lift_plastic_cum = accumulate(obs_lift_plastic, interval)
pred_lift_plastic_bins, pred_lift_plastic_cum = accumulate(pred_lift_plastic, interval)
obs_move_plastic_bins, obs_move_plastic_cum = accumulate(obs_move_plastic, interval)
pred_move_plastic_bins, pred_move_plastic_cum = accumulate(pred_move_plastic, interval)
obs_lift_aluminum_bins, obs_lift_aluminum_cum = accumulate(obs_lift_aluminum, interval)
pred_lift_aluminum_bins, pred_lift_aluminum_cum = accumulate(pred_lift_aluminum, interval)
obs_move_aluminum_bins, obs_move_aluminum_cum = accumulate(obs_move_aluminum, interval)
pred_move_aluminum_bins, pred_move_aluminum_cum = accumulate(pred_move_aluminum, interval)
obs_lift_steel_bins, obs_lift_steel_cum = accumulate(obs_lift_steel, interval)
pred_lift_steel_bins, pred_lift_steel_cum = accumulate(pred_lift_steel, interval)
obs_move_steel_bins, obs_move_steel_cum = accumulate(obs_move_steel, interval)
pred_move_steel_bins, pred_move_steel_cum = accumulate(pred_move_steel, interval)

# Plot the results
def plot_line(obs_bins, obs_cum, pred_bins, pred_cum, title):
    plt.figure(figsize=(10, 6))
    if len(obs_bins) == 0 or len(pred_bins) == 0:
        plt.title(f"No data available for {title}")
        return
    plt.plot(obs_bins, obs_cum, label='Observed', color='purple')
    plt.plot(pred_bins, pred_cum, label='Predicted', color='grey')
    plt.xlabel('Time (s)')
    plt.ylabel('Cumulative Carbon (kg CO2)')
    plt.title(title)
    plt.legend()
    plt.show()

plot_line(obs_lift_plastic_bins, obs_lift_plastic_cum, pred_lift_plastic_bins, pred_lift_plastic_cum, 'Lift Tasks with Plastic')
plot_line(obs_move_plastic_bins, obs_move_plastic_cum, pred_move_plastic_bins, pred_move_plastic_cum, 'Move Tasks with Plastic')
plot_line(obs_lift_aluminum_bins, obs_lift_aluminum_cum, pred_lift_aluminum_bins, pred_lift_aluminum_cum, 'Lift Tasks with Aluminum')
plot_line(obs_move_aluminum_bins, obs_move_aluminum_cum, pred_move_aluminum_bins, pred_move_aluminum_cum, 'Move Tasks with Aluminum')
plot_line(obs_lift_steel_bins, obs_lift_steel_cum, pred_lift_steel_bins, pred_lift_steel_cum, 'Lift Tasks with Steel')
plot_line(obs_move_steel_bins, obs_move_steel_cum, pred_move_steel_bins, pred_move_steel_cum, 'Move Tasks with Steel')

# Efficiency prediction
new_task_time = 10  # time for a new task in seconds
new_robot_id = 1  # id for the new robot
new_task_type = 1  # task type (1 for lift, 0 for move)
new_material_density = 0.9  # material density for the new task
new_material_efficiency = 0.95  # material efficiency for the new task
new_material_embodied_carbon = 6.0  # material embodied carbon for the new task
new_battery_level = 950  # battery level for a new task in watt-hours

# Convert the list to a NumPy array
new_task_data = np.array([[new_task_time, new_robot_id, new_task_type, new_material_density, new_material_efficiency, new_material_embodied_carbon, new_battery_level]])

# Make the prediction
predicted_carbon = model.predict(new_task_data)
print(f"Predicted Carbon Consumption for a {new_task_time}-second task by robot {new_robot_id} with material density {new_material_density}, efficiency {new_material_efficiency}, and embodied carbon {new_material_embodied_carbon}: {predicted_carbon[0][0]:.2f} kg CO2")
