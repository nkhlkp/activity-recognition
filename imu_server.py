import socket
import time
import csv
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.animation as animation

HOST = "10.145.27.140"   # Listen on all interfaces
PORT = 5000        # Must match phone app

SAVE_TO_CSV = True
CSV_FILENAME = "imu_data.csv"

# Plot settings
PLOT_WINDOW = 100  # Number of points to display in the plot
PLOT_UPDATE_RATE = 10  # Update plot every N packets

def start_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((HOST, PORT))
    server.listen(1)

    print(f"[INFO] Listening on {HOST}:{PORT} ...")
    conn, addr = server.accept()
    print(f"[INFO] Connected by {addr}")

    return conn


def setup_plot():
    """Initialize the plot with two subplots for accelerometer and gyroscope."""
    plt.ion()  # Enable interactive mode
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Accelerometer plot
    ax1.set_title("Accelerometer (m/s²)")
    ax1.set_xlabel("Sample")
    ax1.set_ylabel("Acceleration")
    ax1.grid(True)
    line_ax, = ax1.plot([], [], 'r-', label='ax', linewidth=1)
    line_ay, = ax1.plot([], [], 'g-', label='ay', linewidth=1)
    line_az, = ax1.plot([], [], 'b-', label='az', linewidth=1)
    ax1.legend(loc='upper right')
    
    # Gyroscope plot
    ax2.set_title("Gyroscope (rad/s)")
    ax2.set_xlabel("Sample")
    ax2.set_ylabel("Angular Velocity")
    ax2.grid(True)
    line_gx, = ax2.plot([], [], 'r-', label='gx', linewidth=1)
    line_gy, = ax2.plot([], [], 'g-', label='gy', linewidth=1)
    line_gz, = ax2.plot([], [], 'b-', label='gz', linewidth=1)
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    
    return fig, ax1, ax2, (line_ax, line_ay, line_az), (line_gx, line_gy, line_gz)


def update_plot(ax1, ax2, lines_acc, lines_gyro, data_acc, data_gyro):
    """Update the plot with new data."""
    x = list(range(len(data_acc[0])))
    
    # Update accelerometer lines
    lines_acc[0].set_data(x, data_acc[0])
    lines_acc[1].set_data(x, data_acc[1])
    lines_acc[2].set_data(x, data_acc[2])
    
    # Update gyroscope lines
    lines_gyro[0].set_data(x, data_gyro[0])
    lines_gyro[1].set_data(x, data_gyro[1])
    lines_gyro[2].set_data(x, data_gyro[2])
    
    # Rescale axes
    ax1.relim()
    ax1.autoscale_view()
    ax2.relim()
    ax2.autoscale_view()
    
    plt.pause(0.001)  # Short pause to update the plot


def main():
    conn = start_server()
    conn.settimeout(0.05)  # Short timeout so matplotlib stays responsive

    buffer = ""
    packet_count = 0
    start_time = time.time()

    # Initialize data buffers for plotting
    data_acc = [deque(maxlen=PLOT_WINDOW) for _ in range(3)]  # ax, ay, az
    data_gyro = [deque(maxlen=PLOT_WINDOW) for _ in range(3)]  # gx, gy, gz
    
    # Setup plot
    fig, ax1, ax2, lines_acc, lines_gyro = setup_plot()

    if SAVE_TO_CSV:
        csv_file = open(CSV_FILENAME, mode="w", newline="")
        writer = csv.writer(csv_file)
        writer.writerow(["timestamp_ms", "ax", "ay", "az", "gx", "gy", "gz"])

    try:
        while True:
            try:
                data = conn.recv(4096).decode("utf-8")
            except socket.timeout:
                plt.pause(0.01)  # Keep the plot responsive while waiting
                continue

            if not data:
                print("[WARNING] Connection closed by client.")
                break

            buffer += data

            # Process complete lines only
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()

                if not line:
                    continue

                parts = line.split(",")

                # Support both 4-field (accel only) and 7-field (accel+gyro)
                if len(parts) == 4:
                    timestamp_ms = float(parts[0])
                    ax, ay, az = map(float, parts[1:4])
                    gx, gy, gz = 0.0, 0.0, 0.0
                elif len(parts) == 7:
                    timestamp_ms = float(parts[0])
                    ax, ay, az = map(float, parts[1:4])
                    gx, gy, gz = map(float, parts[4:7])
                else:
                    continue  # Skip malformed lines

                packet_count += 1

                # Add data to buffers
                data_acc[0].append(ax)
                data_acc[1].append(ay)
                data_acc[2].append(az)
                data_gyro[0].append(gx)
                data_gyro[1].append(gy)
                data_gyro[2].append(gz)

                # Update plot periodically
                if packet_count % PLOT_UPDATE_RATE == 0:
                    update_plot(ax1, ax2, lines_acc, lines_gyro, data_acc, data_gyro)

                # Print occasionally to avoid flooding terminal
                if packet_count % 50 == 0:
                    elapsed = time.time() - start_time
                    rate = packet_count / elapsed
                    print(f"[INFO] Packets: {packet_count} | Rate: {rate:.2f} Hz")

                if SAVE_TO_CSV:
                    writer.writerow([timestamp_ms, ax, ay, az, gx, gy, gz])

    except KeyboardInterrupt:
        print("\n[INFO] Server stopped manually.")

    finally:
        conn.close()
        if SAVE_TO_CSV:
            csv_file.close()
        plt.close('all')


if __name__ == "__main__":
    main()
