from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

def add_signal_length_to_title(title, signal_time_ms: np.ndarray):
    dur_s = (signal_time_ms.max() - signal_time_ms.min()) / 1000
    plt.title(f"{title} ({dur_s:.2f} seconds)")

def create_figure(title, signal_time_ms: np.ndarray, xlab='', ylab=''):
    plt.figure(figsize=(8, 6))
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.grid(True)
    add_signal_length_to_title(title, signal_time_ms)
    plt.xlim(signal_time_ms.min(), signal_time_ms.max())

def main():
    # Load the accelerometer and gyroscope data from the CSV files
    accelerometer_data = pd.read_csv('stand_1min.csv', sep=';', skiprows=1)
    gyroscope_data = pd.read_csv('stand_1min.csv', sep=';', skiprows=1)

    # Plot accelerometer data
    time_acc_ms = accelerometer_data['Time since start in ms ']
    accelerometer_x = accelerometer_data['ACCELEROMETER X (m/s²)']
    accelerometer_y = accelerometer_data['ACCELEROMETER Y (m/s²)']
    accelerometer_z = accelerometer_data['ACCELEROMETER Z (m/s²)']
    create_figure('Accelerometer', time_acc_ms, 'Time (ms)', 'Acceleration (m/s²)')
    plt.plot(time_acc_ms, accelerometer_x, label='x (m/s²)', color='red', linestyle='-', linewidth=2)
    plt.plot(time_acc_ms, accelerometer_y, label='y (m/s²)', color='green', linestyle='-', linewidth=2)
    plt.plot(time_acc_ms, accelerometer_z, label='z (m/s²)', color='blue', linestyle='-', linewidth=2)
    plt.legend(loc='upper right')
    plt.show()

    # Plot gyroscope data
    time_gyro_ms = accelerometer_data['Time since start in ms ']
    gyroscope_x = gyroscope_data['GYROSCOPE X (rad/s)']
    gyroscope_y = gyroscope_data['GYROSCOPE Y (rad/s)']
    gyroscope_z = gyroscope_data['GYROSCOPE Z (rad/s)']
    create_figure('Gyroscope', time_gyro_ms, 'Time (ms)', 'Angular Velocity (rad/s)')
    plt.plot(time_gyro_ms, gyroscope_x, label='x (rad/s)', color='red', linestyle='-', linewidth=2)
    plt.plot(time_gyro_ms, gyroscope_y, label='y (rad/s)', color='green', linestyle='-', linewidth=2)
    plt.plot(time_gyro_ms, gyroscope_z, label='z (rad/s)', color='blue', linestyle='-', linewidth=2)
    plt.legend(loc='upper right')
    plt.show()


if __name__ == "__main__":
    main()
