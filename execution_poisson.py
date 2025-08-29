import numpy as np
import matplotlib.pyplot as plt
from math import log
import heapq
from multiprocessing import Pool, cpu_count
import time
import tqdm
from functools import partial
from config_poisson import *

class ErlangCSimulation:
    def __init__(self, arrival_rate, service_rate, num_servers, abandonment_rate, sim_time):
        self.arrival_rate = arrival_rate
        self.service_rate = service_rate
        self.num_servers = num_servers
        self.abandonment_rate = abandonment_rate
        self.sim_time = sim_time
        
        # State variables
        self.queue = []
        self.servers_busy = 0
        self.queue_length_history = []
        self.time_history = []
        self.current_time = 0
        self.event_queue = []
        
        # Statistics
        self.total_arrivals = 0
        self.total_served = 0
        self.total_abandoned = 0
    
    def exponential(self, rate):
        import math
        if rate == 0:
            return math.inf
        """Generate exponential random variate."""
        return -log(1 - np.random.random()) / rate
    
    def schedule_event(self, event_time, event_type, customer_id=None):
        heapq.heappush(self.event_queue, (event_time, event_type, customer_id))
    
    def run_simulation(self):
        # Schedule first arrival
        first_arrival = self.exponential(self.arrival_rate)
        self.schedule_event(first_arrival, 'arrival')
        
        self.queue_length_history.append(0)
        self.time_history.append(0)
        
        while self.current_time < self.sim_time:
            if not self.event_queue:
                break
                
            event_time, event_type, customer_id = heapq.heappop(self.event_queue)
            self.current_time = event_time
            
            self.queue_length_history.append(len(self.queue))
            self.time_history.append(self.current_time)
            
            if event_type == 'arrival':
                self.process_arrival()
            elif event_type == 'service':
                self.process_service(customer_id)
            elif event_type == 'abandonment':
                self.process_abandonment(customer_id)
        
        self.queue_length_history.append(len(self.queue))
        self.time_history.append(self.current_time)

        last_length = self.queue_length_history[-1]

        # calculate the number in system at the end of the simulation
        if self.servers_busy > 0:
            last_length += self.servers_busy
        
        return last_length  # Return the last queue length
    
    def process_arrival(self):
        self.total_arrivals += 1
        
        next_arrival = self.current_time + self.exponential(self.arrival_rate)
        self.schedule_event(next_arrival, 'arrival')
        
        if self.servers_busy < self.num_servers:
            self.servers_busy += 1
            service_time = self.current_time + self.exponential(self.service_rate)
            self.schedule_event(service_time, 'service', self.total_arrivals)
        else:
            self.queue.append(self.total_arrivals)
            abandonment_time = self.current_time + self.exponential(self.abandonment_rate)
            self.schedule_event(abandonment_time, 'abandonment', self.total_arrivals)
    
    def process_service(self, customer_id):
        self.total_served += 1
        self.servers_busy -= 1
        
        if self.queue:
            next_customer = self.queue.pop(0)
            service_time = self.current_time + self.exponential(self.service_rate)
            self.schedule_event(service_time, 'service', next_customer)
            self.servers_busy += 1
    
    def process_abandonment(self, customer_id):
        if customer_id in self.queue:
            self.queue.remove(customer_id)
            self.total_abandoned += 1

def run_single_simulation(params):
    """Wrapper function for parallel execution."""
    arrival_rate, service_rate, num_servers, abandonment_rate, sim_time = params
    sim = ErlangCSimulation(arrival_rate, service_rate, num_servers, abandonment_rate, sim_time)
    return sim.run_simulation()

# Prepare parameter combinations
params_list = []
for n in n_values:
    for _ in range(num_paths_queue):
        params_list.append((arrival_rate, service_rate, n, abandonment_rate, sim_time))

# Run simulations in parallel
print(f"Starting simulations with multiple CPUs...")
start_time = time.time()

with Pool(processes=num_of_cpus) as pool:
    # results = pool.map(run_single_simulation, params_list)
    # Using tqdm to show progress
    results = list(tqdm.tqdm(pool.imap(run_single_simulation, params_list), total=len(params_list), desc="Running simulations"))

# save results to a file
np.save('/home/jjiac/jinghan_hkust/Queue/data/poisson_queue_NIS(noaban).npy', results)
print("Queue results (n x num_paths_queue) saved.--------------------------------------------------------------------------------------")

# ################################################################################# Theoretical Results #################################################################################

# def simulate_single_path(args, Y0, lambda_, mu, theta, n, T, dt):
#     """Simulate a single path of the SDE"""
#     seed, _ = args
#     np.random.seed(seed)
    
#     num_steps = int(T / dt)
#     Y = Y0
    
#     def b(x):
#         return lambda_ - mu * np.minimum(x, n) - theta * np.maximum(x - n, 0)
    
#     sqrt_2lambda = np.sqrt(2 * lambda_)
#     dW = np.random.normal(0, np.sqrt(dt), size=num_steps)
    
#     for i in range(num_steps):
#         Y = Y + b(Y) * dt + sqrt_2lambda * dW[i]
    
#     return Y  # Return just the final value

# def parallel_simulation(Y0, lambda_, mu, theta, n, T, dt, num_paths=1000, num_cpus=200):
#     """Parallel simulation returning mean final value"""
#     seeds = np.random.randint(0, 2**32 - 1, size=num_paths)
#     args = list(zip(seeds, range(num_paths)))
    
#     sim_func = partial(simulate_single_path,
#                      Y0=Y0, lambda_=lambda_, mu=mu, theta=theta,
#                      n=n, T=T, dt=dt)
    
#     with Pool(processes=min(num_cpus, cpu_count())) as pool:
#         results = list(tqdm.tqdm(pool.imap(sim_func, args), total=num_paths))
    
#     final_values = np.array(results)
#     return final_values

# theo_array = []

# # Run simulations for each n value
# for n in n_values:
#     print(f"Simulating for n = {n}...")
#     final_values = parallel_simulation(Y0, lambda_, mu, theta, n, T, dt, num_paths_theo, num_cpus)
#     theo_array.append(final_values)

# # Convert to numpy array for easier handling
# theo_array = np.array(theo_array)

# # save theoretical results
# np.save('/home/jjiac/jinghan_hkust/Queue/data/poisson_theo_NIS.npy', theo_array)
# print("Theoretical results (n x num_paths_theo) saved.--------------------------------------------------------------------------------------")