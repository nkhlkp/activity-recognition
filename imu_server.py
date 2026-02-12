import socket
import time
import csv
from collections import deque
import matplotlib.pyplot as plt

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
    """Initialize the plot for accelerometer data."""
    plt.ion()  # Enable interactive mode
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.set_title("Accelerometer (m/s²)")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Acceleration")
    ax.grid(True)
    line_ax, = ax.plot([], [], 'r-', label='ax', linewidth=1)
    line_ay, = ax.plot([], [], 'g-', label='ay', linewidth=1)
    line_az, = ax.plot([], [], 'b-', label='az', linewidth=1)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    return fig, ax, (line_ax, line_ay, line_az)


def update_plot(ax, lines_acc, data_acc):
    """Update the plot with new data."""
    x = list(range(len(data_acc[0])))
    
    lines_acc[0].set_data(x, data_acc[0])
    lines_acc[1].set_data(x, data_acc[1])
    lines_acc[2].set_data(x, data_acc[2])
    
    ax.relim()
    ax.autoscale_view()
    
    plt.pause(0.001)  # Short pause to update the plot


def main():
    conn = start_server()
    conn.settimeout(0.05)  # Short timeout so matplotlib stays responsive

    buffer = ""
    packet_count = 0
    start_time = time.time()

    # Initialize data buffers for plotting
    data_acc = [deque(maxlen=PLOT_WINDOW) for _ in range(3)]  # ax, ay, az
    
    # Setup plot
    fig, ax, lines_acc = setup_plot()

    if SAVE_TO_CSV:
        csv_file = open(CSV_FILENAME, mode="w", newline="")
        writer = csv.writer(csv_file)
        writer.writerow(["timestamp_ms", "ax", "ay", "az"])

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

                if len(parts) != 4:
                    continue  # Skip malformed lines

                timestamp_ms = float(parts[0])
                ax, ay, az = map(float, parts[1:4])

                packet_count += 1

                # Add data to buffers
                data_acc[0].append(ax)
                data_acc[1].append(ay)
                data_acc[2].append(az)

                # Update plot periodically
                if packet_count % PLOT_UPDATE_RATE == 0:
                    update_plot(ax, lines_acc, data_acc)

                # Print occasionally to avoid flooding terminal
                if packet_count % 50 == 0:
                    elapsed = time.time() - start_time
                    rate = packet_count / elapsed
                    print(f"[INFO] Packets: {packet_count} | Rate: {rate:.2f} Hz")

                if SAVE_TO_CSV:
                    writer.writerow([timestamp_ms, ax, ay, az])

    except KeyboardInterrupt:
        print("\n[INFO] Server stopped manually.")

    finally:
        conn.close()
        if SAVE_TO_CSV:
            csv_file.close()
        plt.close('all')


if __name__ == "__main__":
    main()
