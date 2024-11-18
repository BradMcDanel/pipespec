import time
import torch.multiprocessing as mp
import pynvml

class GPUMonitor:
    def __init__(self, interval=0.1):
        self.interval = interval
        self.stop_event = mp.Event()
        self.queue = mp.Queue(maxsize=100000)  # Set a max size to prevent memory issues
        self.results = []
        self.process = None
        
    def start(self):
        if self.process is not None and self.process.is_alive():
            print("Monitoring is already running.")
            return
        self.stop_event.clear()
        self.process = mp.Process(target=self._monitor_gpu)
        self.process.daemon = True  # Make it a daemon process
        self.process.start()
        
    def stop(self):
        if self.process is None or not self.process.is_alive():
            print("Monitoring is not running.")
            return
        
        self.stop_event.set()
        # Only wait for a short time to avoid hanging
        self.process.join(timeout=5.0)
        
        if self.process.is_alive():
            print("GPU monitoring process did not terminate normally, terminating forcefully")
            self.process.terminate()
            self.process.join(timeout=1.0)
            
            if self.process.is_alive():
                self.process.kill()
                
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
                    
                try:
                    # Use put_nowait to avoid blocking
                    self.queue.put_nowait({
                        'timestamp': time.time() - start_time,
                        'gpu_utilizations': gpu_utilizations,
                        'gpu_memories': gpu_memories,
                        'gpu_powers': gpu_powers
                    })
                except:
                    # Queue is full, skip this sample
                    pass
                    
                time.sleep(self.interval)
                
        finally:
            pynvml.nvmlShutdown()
            
    def collect_results(self):
        while True:
            try:
                self.results.append(self.queue.get_nowait())
            except:
                break
