import numpy as np
import matplotlib.pyplot as plt
from math import log
import heapq
from multiprocessing import Pool, cpu_count
import time
import tqdm
from functools import partial
from config_dspp import *

class ErlangCSimulation:
    def __init__(self, arrival_rate, service_rate, num_servers, abandonment_rate, sim_time, kappa=1.0, sigma=0.5, alpha=0.5):
        self.arrival_rate = arrival_rate  # lambda
        self.service_rate = service_rate  # mu
        self.num_servers = num_servers
        self.abandonment_rate = abandonment_rate  # gamma
        self.sim_time = sim_time
        self.kappa = kappa
        self.sigma = sigma
        self.alpha = alpha
        
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
        
        # DSPP setup ##########################################################################
        self.dt = 0.01  # Time step for SDE
        self.time_steps = np.arange(0, self.sim_time + self.dt, self.dt)
        self.rate_trajectory = self.simulate_rate_trajectory()
        self.lambda_max = 3 * self.arrival_rate  # Upper bound for thinning
        # DSPP setup ##########################################################################


    def simulate_rate_trajectory(self):
        """Simulate X(t) using Euler-Maruyama method for the SDE."""
        X = np.zeros(len(self.time_steps))
        a = 2 * self.kappa / (self.sigma**2 * self.arrival_rate**(self.alpha - 1))
        b = 2 * self.kappa / (self.sigma**2 * self.arrival_rate**self.alpha)
        X[0] = np.random.gamma(a, 1/b)  # Scale = 1/rate for NumPy
        # X[]0] = self.arrival_rate  # Initial rate
        for i in range(1, len(self.time_steps)):
            X_prev = X[i-1]
            drift = self.kappa * (self.arrival_rate - X_prev) * self.dt
            diffusion = self.sigma * np.sqrt(self.arrival_rate ** self.alpha * X_prev) * np.sqrt(self.dt) * np.random.normal()
            X[i] = max(X_prev + drift + diffusion, 0)  # Ensure non-negative
        return X

    def get_rate_at_time(self, t):
        """Interpolate X(t) at time t."""
        idx = int(t / self.dt)
        if idx >= len(self.rate_trajectory) - 1:
            return self.rate_trajectory[-1]
        frac = (t - idx * self.dt) / self.dt
        return self.rate_trajectory[idx] * (1 - frac) + self.rate_trajectory[idx + 1] * frac

    def generate_next_arrival(self):
        """Generate next arrival time using thinning for non-homogeneous Poisson process."""
        t = self.current_time
        while True:
            t += -log(1 - np.random.random()) / self.lambda_max
            if t > self.sim_time:
                return self.sim_time + 1  # No more arrivals
            rate = self.get_rate_at_time(t)
            if np.random.random() < rate / self.lambda_max:
                return t
        return self.sim_time + 1

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
        first_arrival = self.generate_next_arrival()
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
        
        # # Calculate time-weighted average queue length
        # time_intervals = np.diff(self.time_history)
        # queue_lengths = np.array(self.queue_length_history[:-1])
        # avg_queue_length = np.sum(time_intervals * queue_lengths) / self.sim_time

        last_length = self.queue_length_history[-1]

        # calculate the number in system at the end of the simulation
        if self.servers_busy > 0:
            last_length += self.servers_busy
        
        return last_length  # Return the last queue length

    def process_arrival(self):
        self.total_arrivals += 1
        
        # Schedule next arrival using DSPP
        next_arrival = self.generate_next_arrival()
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
    arrival_rate, service_rate, num_servers, abandonment_rate, sim_time, kappa, sigma, alpha = params
    sim = ErlangCSimulation(arrival_rate, service_rate, num_servers, abandonment_rate, sim_time, kappa, sigma, alpha)
    return sim.run_simulation()


# Prepare parameter combinations
params_list = []
for n in n_values:
    for _ in range(num_paths_queue):
        params_list.append((arrival_rate, service_rate, n, abandonment_rate, sim_time, kappa, sigma, alpha))

# Run simulations in parallel
print(f"Starting simulations with multiple CPUs...")
start_time = time.time()

with Pool(processes=num_cpus) as pool:
    results = list(tqdm.tqdm(pool.imap(run_single_simulation, params_list), total=len(params_list), desc="Running simulations"))

# save results to a file
np.save(f'/home/jjiac/jinghan_hkust/Queue/data/dspp_queue_NIS(noaban)_alpha{alpha}.npy', np.array(results))
print("Queue results (n x num_paths_queue) saved.--------------------------------------------------------------------------------------")

# ################################################################################# Theoretical Results #################################################################################
# def simulate_single_path(args, params):
#     """Simulate a single path of Y and U."""
#     seed, _ = args
#     np.random.seed(seed)
    
#     Y0 = params['Y0']
#     lambda_ = params['lambda_']
#     alpha = params['alpha']
#     mu = params['mu']
#     theta = params['theta']
#     n = params['n']
#     T = params['T']
#     dt = params['dt']
#     U0 = params['U0']
#     kappa = params['kappa']
#     phi = params['phi']
#     sigma = params['sigma']
    
#     num_steps = int(T / dt)
#     t = np.linspace(0, T, num_steps + 1)
#     Y = np.zeros(num_steps + 1)
#     U = np.zeros(num_steps + 1)
#     Y[0] = Y0
#     U[0] = U0
    
#     def b(x):
#         return lambda_ - mu * np.minimum(x, n) - theta * np.maximum(x - n, 0)
    
#     sqrt_2lambda = (lambda_) ** ((alpha + 1) / 2)  # Corrected to include factor of 2
#     dW_u = np.random.normal(0, np.sqrt(dt), size=num_steps)
#     dW = np.random.normal(0, np.sqrt(dt), size=num_steps)
    
#     for i in range(num_steps):
#         U[i + 1] = U[i] + kappa * (phi - U[i]) * dt + sigma * dW_u[i]
#         # Y[i + 1] = Y[i] + b(Y[i]) * dt + sqrt_2lambda * U[i] * dt + np.sqrt(lambda_) * dW[i]
#         Y[i + 1] = Y[i] + b(Y[i]) * dt + sqrt_2lambda * U[i] * dt + sqrt_2lambda * dW[i]
#         Y[i + 1] = max(Y[i + 1], 0)  # Apply transformation at each step
    
#     return t, Y, U

# def parallel_simulation(Y0, lambda_, alpha, mu, theta, n, T, dt, U0, kappa, phi, sigma, num_paths=1000, num_cpus=20):
#     """Run parallel simulations and return mean and std of final Y values."""
#     seeds = np.random.randint(0, 2**32 - 1, size=num_paths)
#     args = list(zip(seeds, range(num_paths)))
    
#     sim_func = partial(simulate_single_path, params={
#         'Y0': Y0, 'lambda_': lambda_, 'alpha': alpha, 'mu': mu, 'theta': theta,
#         'n': n, 'T': T, 'dt': dt, 'U0': U0, 'kappa': kappa, 'phi': phi, 'sigma': sigma
#     })
    
#     with Pool(processes=min(num_cpus, cpu_count())) as pool:
#         results = list(tqdm.tqdm(pool.imap(sim_func, args), total=num_paths, desc=f"Simulating n={n:.1f}"))
    
#     # Extract final Y values from each path
#     final_values = np.array([Y[-1] for t, Y, U in results])
#     return final_values

# def run_simulation_for_n(n_value, params, num_paths=100, num_cpus=20):
#     """Run simulation for a specific n value and return converged Y statistics."""
#     final_values = parallel_simulation(
#         Y0=params['Y0'], lambda_=params['lambda_'], alpha=params['alpha'], 
#         mu=params['mu'], theta=params['theta'], n=n_value, T=params['T'], 
#         dt=params['dt'], U0=params['U0'], kappa=params['kappa'], 
#         phi=params['phi'], sigma=params['sigma'],
#         num_paths=num_paths, num_cpus=num_cpus
#     )
    
#     return final_values

# # Parameters
# params = {
#     'Y0': Y0,
#     'lambda_': arrival_rate,
#     'alpha': alpha,
#     'mu': service_rate,
#     'theta': abandonment_rate,
#     'T': sim_time,
#     'dt': dt,
#     'U0': U0,
#     'kappa': kappa,
#     'phi': phi,
#     'sigma': sigma
# }

# # Store results
# theo_array = []

# # Run simulations for each n
# for n in tqdm.tqdm(n_values, desc="Processing n values"):
#     final_values = run_simulation_for_n(n, params, num_paths_theo, num_cpus)
#     theo_array.append(final_values)

# # Save theoretical results
# theo_array = np.array(theo_array)
# np.save(f'/home/jjiac/jinghan_hkust/Queue/data/dspp_theo(newcoef)_NIS_alpha{alpha}.npy', theo_array)
# print("Theoretical results (n x num_paths_theo) saved.--------------------------------------------------------------------------------------")