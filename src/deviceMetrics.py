import matplotlib.pyplot as plt

#metrics data for each device
metrics = {
    "Jetson Orin NX": {
        "train_time":        0.10,
        "inference_latency":  6.57,
        "retrain_time":       1.00,
        "cpu_util":          86.0,
        "mem_usage":         15.6,
        "net_recv":          59.0,
        "net_send":           0.0,
        "power":             6.805
    },
    "Xilinx Kria KV260": {
        "train_time":        0.68,
        "inference_latency":21.29,
        "retrain_time":      4.03,
        "cpu_util":         100.0,
        "mem_usage":         32.5,
        "net_recv":           0.0,
        "net_send":           0.0,
        "power":             5.5
    },
    "Raspberry Pi 5": {
        "train_time":        0.13,
        "inference_latency": 9.26,
        "retrain_time":      0.69,
        "cpu_util":         100.0,
        "mem_usage":         15.9,
        "net_recv":         342.0,
        "net_send":         132.0,
        "power":             5.5
    },
    "Raspberry Pi Zero W": {
        "train_time":        1.19,
        "inference_latency":106.71,
        "retrain_time":      4.67,
        "cpu_util":          95.0,
        "mem_usage":         51.0,
        "net_recv":           0.0,
        "net_send":           0.0,
        "power":             0.8
    }
}

#assign a unique color to each device
color_map = {
    "Jetson Orin NX":     "#1f77b4",  # blue
    "Xilinx Kria KV260":  "#ff7f0e",  # orange
    "Raspberry Pi 5":     "#2ca02c",  # green
    "Raspberry Pi Zero W":"#d62728"   # red
}

devices = list(metrics.keys())
colors  = [color_map[d] for d in devices]

def vals(key):
    """Helper to extract a list of metric values in device order."""
    return [metrics[d][key] for d in devices]


#training Time per Device
plt.figure()
plt.bar(devices, vals("train_time"), color=colors)
plt.title("Training Time per Device")
plt.ylabel("Time (s)")

#inference Latency per Device
plt.figure()
plt.bar(devices, vals("inference_latency"), color=colors)
plt.title("Inference Latency per Device")
plt.ylabel("Latency (ms)")

#retraining Time per Device
plt.figure()
plt.bar(devices, vals("retrain_time"), color=colors)
plt.title("Retraining Time per Device")
plt.ylabel("Time (s)")

#CPU Utilization during Training
plt.figure()
plt.bar(devices, vals("cpu_util"), color=colors)
plt.title("CPU Utilization during Training")
plt.ylabel("CPU Utilization (%)")

#memory Usage during Training
plt.figure()
plt.bar(devices, vals("mem_usage"), color=colors)
plt.title("Memory Usage during Training")
plt.ylabel("Memory Usage (% of total RAM)")

#power Consumption during Training
plt.figure()
plt.bar(devices, vals("power"), color=colors)
plt.title("Power Consumption during Training")
plt.ylabel("Power (W)")

plt.tight_layout()
plt.show()