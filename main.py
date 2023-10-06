from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from scipy.io import wavfile
import sounddevice as sd
import scipy.io as sio
import pandas as pd
import numpy as np
import os

def add_signal_length_to_title(title, signal_time_ms: np.ndarray, seconds=False):
    dur_s = signal_time_ms.max() - signal_time_ms.min()
    if not seconds:  # then miliseconds
        dur_s /= 1000
    plt.title(f"{title} ({dur_s:.2f} seconds)")

def create_figure(title, signal_time_ms: np.ndarray, xlab='', ylab='', seconds=False):
    plt.figure(figsize=(8, 6))
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.grid(True)
    add_signal_length_to_title(title, signal_time_ms, seconds)
    plt.xlim(signal_time_ms.min(), signal_time_ms.max())

def task_1():
    accelerometer_data = pd.read_csv("stand_1min.csv", sep=';', skiprows=1)
    gyroscope_data = pd.read_csv("stand_1min.csv", sep=';', skiprows=1)

    time_acc_ms = accelerometer_data["Time since start in ms "]
    accelerometer_x = accelerometer_data["ACCELEROMETER X (m/s²)"]
    accelerometer_y = accelerometer_data["ACCELEROMETER Y (m/s²)"]
    accelerometer_z = accelerometer_data["ACCELEROMETER Z (m/s²)"]
    create_figure("Accelerometer", time_acc_ms, "Time (ms)", "Acceleration (m/s²)")
    plt.plot(time_acc_ms, accelerometer_x, label="x (m/s²)", color="red", linestyle='-', linewidth=2)
    plt.plot(time_acc_ms, accelerometer_y, label="y (m/s²)", color="green", linestyle='-', linewidth=2)
    plt.plot(time_acc_ms, accelerometer_z, label="z (m/s²)", color="blue", linestyle='-', linewidth=2)
    plt.legend(loc="upper right")
    plt.savefig("Accelerometer.png")
    plt.show()

    time_gyro_ms = accelerometer_data["Time since start in ms "]
    gyroscope_x = gyroscope_data["GYROSCOPE X (rad/s)"]
    gyroscope_y = gyroscope_data["GYROSCOPE Y (rad/s)"]
    gyroscope_z = gyroscope_data["GYROSCOPE Z (rad/s)"]
    create_figure("Gyroscope", time_gyro_ms, "Time (ms)", "Angular Velocity (rad/s)")
    plt.plot(time_gyro_ms, gyroscope_x, label="x (rad/s)", color="red", linestyle='-', linewidth=2)
    plt.plot(time_gyro_ms, gyroscope_y, label="y (rad/s)", color="green", linestyle='-', linewidth=2)
    plt.plot(time_gyro_ms, gyroscope_z, label="z (rad/s)", color="blue", linestyle='-', linewidth=2)
    plt.legend(loc="upper right")
    plt.savefig("Gyroscope.png")
    plt.show()

def record_audio(duration, sampling_rate):
    print(f"Recording audio at {sampling_rate} Hz ({duration:.2f}s)...")
    audio_data = sd.rec(int(duration * sampling_rate), samplerate=sampling_rate, channels=1, dtype='float64')
    sd.wait()
    return audio_data

def save_audio(filename, audio_data, sampling_rate):
    print(f"Saving audio to {filename}...")
    wavfile.write(filename, sampling_rate, audio_data)

def plot_single_signal(data, sampling_rate, title, ylab="Amplitude", seconds=False):
    if seconds:
        time_ms, xlab = np.linspace(0, len(data) / sampling_rate, num=len(data)), "Time (s)"
    else:
        time_ms, xlab = np.linspace(0, len(data) / sampling_rate * 1000, num=len(data)), "Time (ms)"
    create_figure(title, time_ms, xlab, ylab, seconds)
    plt.plot(time_ms, data)
    plt.savefig(f"{title}.png")
    plt.show()

def task_2():
    duration = 5

    sampling_rate_8k = 8000
    audio_data_8k = record_audio(duration, sampling_rate_8k)
    save_audio("audio_8k.wav", audio_data_8k, sampling_rate_8k)
    plot_single_signal(audio_data_8k, sampling_rate_8k, "Audio at 8 kHz")

    sampling_rate_44k = 44100
    audio_data_44k = record_audio(duration, sampling_rate_44k)
    save_audio("audio_44k.wav", audio_data_44k, sampling_rate_44k)
    plot_single_signal(audio_data_44k, sampling_rate_44k, "Audio at 44.1 kHz")

def load_eeg_signal(file_path):
    mat_data = sio.loadmat(file_path)
    eeg_signal = mat_data["sig"][0]
    return eeg_signal

def task_3():
    eeg_healthy = load_eeg_signal("EEG_healthy/eeg_healthy_1.mat")
    eeg_sick = load_eeg_signal("EEG_sick/eeg_sick_1.mat")

    sampling_rate = 256
    plot_single_signal(eeg_healthy, sampling_rate, "EEG Signal (Healthy)")
    plot_single_signal(eeg_sick, sampling_rate, "EEG Signal (Sick)")

    np.save("eeg_healthy_signal", eeg_healthy)
    np.save("eeg_sick_signal", eeg_sick)

def load_and_plot_ecg(file_path, title):
    with np.load(file_path) as data:
        signal = data["signal"]
        labels = data["labels"]
        labels_indexes = data["labels_indexes"]
        sampling_rate = data["fs"]
        units = data["units"]

    time_ms = np.linspace(0, signal.shape[0] / sampling_rate * 1000, num=signal.shape[0])
    create_figure(title, time_ms, "Time (ms)", f"Amplitude ({units})")
    plt.plot(time_ms, signal)
    for idx, label in zip(labels_indexes, labels):
        plt.annotate(label, (time_ms[idx], signal[idx]), textcoords="offset points", xytext=(0,10), ha="center")
    plt.savefig(f"{title}.png")
    plt.show()

def task_4():
    norm_dir = "./norm"
    anomaly_dir = "./anomaly"
    norm_files = os.listdir(norm_dir)
    anomaly_files = os.listdir(anomaly_dir)
    first_norm_file = os.path.join(norm_dir, norm_files[0])
    first_anomaly_file = os.path.join(anomaly_dir, anomaly_files[0])
    load_and_plot_ecg(first_norm_file, "Normal ECG")
    load_and_plot_ecg(first_anomaly_file, "Anomalous ECG")

def process_heart_signal(signal_periods_ms, title):
    signal_time_ms = np.cumsum(signal_periods_ms)
    # convert to Hz for every time step
    signal_freq = 1000 / signal_periods_ms

    interpolation_function = interp1d(signal_time_ms, signal_freq, kind='linear', fill_value='extrapolate')
    new_time_ms = np.arange(signal_time_ms.min(), signal_time_ms.max() + 1000, step=1000)

    interpolated_freq = interpolation_function(new_time_ms)

    new_time_s = new_time_ms / 1000

    create_figure(title, new_time_s, "Time (s)", "Frequency (Hz)", seconds=True)
    plt.plot(new_time_s, interpolated_freq)
    plt.savefig(f"{title}.png")
    plt.show()

    np.save(f"{title}.npy", interpolated_freq)

def task_5():
    healthy_signal_periods_ms = sio.loadmat("heart_rate_norm.mat")["hr_norm"].T[0][1:]
    sick_signal_periods_ms = sio.loadmat("heart_rate_apnea.mat")["hr_ap"].T[0][1:]

    process_heart_signal(healthy_signal_periods_ms, "Healthy Heart Rate Signal")
    process_heart_signal(sick_signal_periods_ms, "Sick Heart Rate Signal")

def task_6():
    data_dir = "./cop_data/data"
    all_columnns = ["time_ms", "top_left_f_kg", "top_right_f_kg", "bottom_left_f_kg", "bottom_right_f_kg", "cop_x", "cop_y", "total_f"]
    selected_columns = ["cop_x", "cop_y"]

    for signal_type in ["base_open", "base_close"]:
        stats_dict = {}

        for athlete_type in ["handball", "acrobats"]:
            athlete_data = pd.DataFrame(columns=selected_columns)
            signal_dir = os.path.join(data_dir, athlete_type, signal_type)
            signal_files = os.listdir(signal_dir)
            file_name = signal_files[0]  # checking the first file for stats
            file_path = os.path.join(signal_dir, file_name)
            signal_data = pd.read_csv(file_path, sep=r'\s+', header=0, names=all_columnns)[selected_columns]
            athlete_data = pd.concat([athlete_data, signal_data])

            plt.figure(figsize=(8, 6))
            plt.plot(athlete_data['cop_x'], athlete_data['cop_y'], linestyle='-', color='blue')
            title = f'Center of Pressure (CoP) ({athlete_type}, {signal_type})'
            plt.title(title)
            plt.xlabel('CoP X-axis (mm)')
            plt.ylabel('CoP Y-axis (mm)')
            plt.grid(True)
            plt.savefig(f"{title}.png")
            plt.show()

            stats_athlete_type = {}
            for postfix, name in [['_X', 'cop_x'], ['_Y', 'cop_y']]:
                stats_athlete_type.update({
                    f"Mean{postfix}": np.mean(athlete_data[name]),
                    f"Median{postfix}": np.median(athlete_data[name]),
                    f"Std{postfix}": np.std(athlete_data[name])
                })
            
            stats_dict[athlete_type] = stats_athlete_type
        
        print(f"Signal: {signal_type}")
        print(pd.DataFrame(stats_dict))
        print()

def plot_spo2_and_hr(time, hr, spo2, filename, postfix=''):
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time, hr, color='blue')
    plt.xlabel('Time (s)')
    plt.ylabel('Heart Rate (bpm)')
    plt.title(f'Heart Rate Signal{postfix}')
    plt.grid(True)
    plt.xlim(time.min(), time.max())

    plt.subplot(2, 1, 2)
    plt.plot(time, spo2, color='green')
    plt.xlabel('Time (s)')
    plt.ylabel('SpO2 (%)')
    plt.title(f'SpO2 Signal{postfix}')
    plt.grid(True)
    plt.xlim(time.min(), time.max())

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

def moving_average_no_overlap(data, window_size):
    data_mat = data.values[:(len(data) // window_size) * window_size].reshape(-1, window_size)
    return np.mean(data_mat, axis=1)

def task_7():
    data_dir = 'spo2_hr_signals/signals/'

    filename = os.listdir(data_dir)[0]
    df = pd.read_csv(os.path.join(data_dir, filename))

    plot_spo2_and_hr(df['Elapsed time(seconds)'], df['hr (bpm)'], df['SpO2(%)'], f'{filename[:-4]}_signals.png', postfix=f"({df['Elapsed time(seconds)'].values[-1]} seconds)")

    time = df['Elapsed time(seconds)'].values[30::30]
    hr_ravg = moving_average_no_overlap(df['hr (bpm)'], 30)
    spo2_ravg = moving_average_no_overlap(df['SpO2(%)'], 30)
    plot_spo2_and_hr(time, hr_ravg, spo2_ravg, f'{filename[:-4]}_avg_signals.png', postfix=f" (30 second window) ({time[-1]} seconds)")

def task_8():
    icp_data = np.loadtxt('TBI_ICP.txt')
    plot_single_signal(icp_data, 125, 'Intracranial Pressure (ICP) Signal', seconds=True, ylab="Amplitude (mmHg)")
    np.save("TBI_ICP.npy", icp_data)

def extract_signal_segment(
        start_time_sec: np.dtype('float64'),
        end_time_sec: np.dtype('float64'),
        sample_rate: np.dtype('int64'),
        signal_vector: np.ndarray
    ):
    """
    Extract a segment of a signal based on start and end times in seconds.
    The boundaries are included.

    Args:
        start_time_sec (float): Start time in seconds.
        end_time_sec (float): End time in seconds.
        sample_rate (int): Sample rate in Hz,
        signal_vector (numpy.ndarray): Vector of signal values.

    Returns:
        extracted_time (numpy.ndarray): Time points of the extracted segment.
        extracted_signal (numpy.ndarray): Signal values of the extracted segment.

    Raises:
        ValueError: If start_time_sec or end_time_sec are outside the time range.
    """
    if start_time_sec < 0 or end_time_sec > len(signal_vector) / sample_rate:
        raise ValueError("Start time or end time is outside the time range.")
    
    time_vector = np.linspace(0, len(signal_vector) / sample_rate, len(signal_vector))

    # first begger and last smaller
    start_index = np.argmax(time_vector >= start_time_sec)
    end_index = np.argmin(time_vector <= end_time_sec)
    
    # last occurence of max value
    while (end_index+1) < len(time_vector) and time_vector[end_index] == time_vector[end_index+1]:
        end_index += 1

    extracted_time = time_vector[start_index:end_index + 1]
    extracted_signal = signal_vector[start_index:end_index + 1]

    return extracted_time, extracted_signal

def task_9():
    sample_rate = 125
    signal_duration = 10
    time_vector = np.linspace(0, signal_duration, sample_rate * signal_duration)
    signal_vector = np.sin(2 * np.pi * time_vector)
    create_figure("Initial Signal", time_vector, xlab="Time (s)", seconds=True)
    plt.plot(time_vector, signal_vector)
    plt.savefig("initial signal.png")
    plt.show()
    try:
        extracted_time, extracted_signal = extract_signal_segment(
            2, 6, sample_rate, signal_vector
        )
        create_figure("Extracted Signal Segment", extracted_time, xlab="Time (s)", seconds=True)
        plt.plot(extracted_time, extracted_signal)
        plt.savefig("extracted signal.png")
        plt.show()
    except ValueError as e:
        print(f"Error: {e}")

def main():
    task_1()
    task_2()
    task_3()
    task_4()
    task_5()
    task_6()
    task_7()
    task_8()
    task_9()

if __name__ == "__main__":
    main()
