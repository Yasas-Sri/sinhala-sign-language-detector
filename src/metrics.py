import time
import psutil
import threading
from prometheus_client import start_http_server, Summary, Histogram, Counter, Gauge

import time
import psutil
import threading
from prometheus_client import start_http_server, Summary, Histogram, Counter, Gauge
from prometheus_client import REGISTRY

# We need to unregister old metrics if Streamlit reloads this file
# Otherwise Prometheus client throws a ValueError for duplicate metrics
_metric_names = [
    'model_inference_latency_seconds',
    'total_predictions_made',
    'system_cpu_usage_percent',
    'system_memory_usage_percent'
]

# Unregister any existing metrics with these names to prevent Duplicate error
for collector in list(REGISTRY._collector_to_names.keys()):
    if any(name in REGISTRY._collector_to_names[collector] for name in _metric_names):
        try:
            REGISTRY.unregister(collector)
        except KeyError:
            pass

# Define Prometheus metrics
INFERENCE_LATENCY = Histogram(
    'model_inference_latency_seconds', 
    'Time spent running the PyTorch hybrid model'
)

TOTAL_PREDICTIONS = Counter(
    'total_predictions_made', 
    'Total number of sign language predictions made', 
    ['sign_class']
)

CPU_USAGE = Gauge(
    'system_cpu_usage_percent', 
    'Current CPU utilization percentage'
)

MEMORY_USAGE = Gauge(
    'system_memory_usage_percent', 
    'Current Memory utilization percentage'
)

def update_system_metrics_loop():
    """Background thread to continuously update CPU and Memory metrics."""
    while True:
        CPU_USAGE.set(psutil.cpu_percent(interval=1))
        MEMORY_USAGE.set(psutil.virtual_memory().percent)
        time.sleep(5)

def start_metrics_server(port=8000):
    """Start the Prometheus metrics server on the specified port."""
    start_http_server(port)
    print(f"Monitoring metrics server started on port {port}")
    
    
    t = threading.Thread(target=update_system_metrics_loop, daemon=True)
    t.start()
