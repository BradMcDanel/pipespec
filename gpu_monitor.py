import time
import torch.multiprocessing as mp
import pynvml

class GPUMonitor:
    def __init__(self, interval=0.5):
        self.interval = interval
        self.stop_event = mp.Event()
        self.queue = mp.Queue()
        self.results = []
        self.process = None

    def start(self):
        if self.process is not None and self.process.is_alive():
            print("Monitoring is already running.")
            return
        self.stop_event.clear()
        self.process = mp.Process(target=self._monitor_gpu)
        self.process.start()

    def stop(self):
        if self.process is None or not self.process.is_alive():
            print("Monitoring is not running.")
            return
        self.stop_event.set()
        self.process.join()
        self.collect_results()

    def exit(self):
        self.stop()

    def clear(self):
        self.results.clear()

    def get_results(self):
        return self.results[:]

    def _monitor_gpu(self):
        pynvml.nvmlInit()
        num_gpus = pynvml.nvmlDeviceGetCount()
        handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(num_gpus)]
        start_time = time.time()
        try:
            while not self.stop_event.is_set():
                gpu_utilizations = []
                gpu_memories = []
                gpu_powers = []
                for handle in handles:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert mW to W
                    gpu_utilizations.append(util.gpu)
                    gpu_memories.append(mem_info.used / mem_info.total * 100)
                    gpu_powers.append(power)
                self.queue.put({
                    'timestamp': time.time() - start_time,
                    'gpu_utilizations': gpu_utilizations,
                    'gpu_memories': gpu_memories,
                    'gpu_powers': gpu_powers
                })
                time.sleep(self.interval)
        finally:
            pynvml.nvmlShutdown()

    def collect_results(self):
        while not self.queue.empty():
            self.results.append(self.queue.get())

