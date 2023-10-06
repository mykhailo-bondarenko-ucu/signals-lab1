from matplotlib import pyplot as plt
from scipy.io import wavfile
import sounddevice as sd
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

def task_1():
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
    plt.savefig('Accelerometer.png')
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
    plt.savefig('Gyroscope.png')
    plt.show()

def record_audio(duration, sampling_rate):
    print(f"Recording audio at {sampling_rate} Hz ({duration:.2f}s)...")
    audio_data = sd.rec(int(duration * sampling_rate), samplerate=sampling_rate, channels=1, dtype='float64')
    sd.wait()
    return audio_data

def save_audio(filename, audio_data, sampling_rate):
    print(f"Saving audio to {filename}...")
    wavfile.write(filename, sampling_rate, audio_data)

def plot_audio(audio_data, sampling_rate, title):
    audio_time_ms = np.linspace(0, len(audio_data) / sampling_rate * 1000, num=len(audio_data))
    create_figure(title, audio_time_ms, 'Time (s)', 'Amplitude')
    plt.plot(audio_time_ms, audio_data)
    plt.savefig(f'{title}.png')
    plt.show()

def task_2():
    duration = 5

    sampling_rate_8k = 8000
    audio_data_8k = record_audio(duration, sampling_rate_8k)
    save_audio('audio_8k.wav', audio_data_8k, sampling_rate_8k)
    plot_audio(audio_data_8k, sampling_rate_8k, 'Audio at 8 kHz')

    sampling_rate_44k = 44100
    audio_data_44k = record_audio(duration, sampling_rate_44k)
    save_audio('audio_44k.wav', audio_data_44k, sampling_rate_44k)
    plot_audio(audio_data_44k, sampling_rate_44k, 'Audio at 44.1 kHz')

def main():
    task_1()
    task_2()

if __name__ == "__main__":
    main()
