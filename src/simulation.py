import numpy as np
import constant as ct

def g(h, alpha=ct.alpha, beta=ct.beta):
    '''This function calculates the value of the transfer function g(h) of Poisson neurons given the values of h, alpha, and beta.
    
    Parameters:
    h (np.ndarray): The potential of every neuron
    alpha (float): The value of alpha used in the transfer function.
    beta (float): The value of beta used in the transfer function.
    '''

    return 1/(1 + np.exp(-2 * alpha * (h - beta)))

def bins_spike(spikes, bins, delta_t=ct.delta_t, N=ct.N, mean=False):
    """
    Bins the spike data into specified time intervals and calculates the sum or mean of spikes within each bin.

    Parameters:
    spikes (ndarray): Array of spike data.
    bins (float): Size of each time bin in milliseconds.
    delta_t (float, optional): Time step size. Defaults to ct.delta_t.
    N (int, optional): Number of neurons. Defaults to ct.N.
    mean (bool, optional): If True, calculates the mean of spikes within each bin. If False, calculates the sum. Defaults to False.

    Returns:
    ndarray: Binned spike data.

    """
    bins_size = int(bins / delta_t)
    spikes = spikes.reshape((int(spikes.shape[0] // bins_size), bins_size, N))
    if mean == True:
        return np.mean(spikes, axis=1)
    else:
        return np.sum(spikes, axis=1)

def mean_spike(spikes, bins, delta_t=ct.delta_t ,N=ct.N):
    spikes = bins_spike(spikes, bins, delta_t, N, True)
    mean_spikes = np.mean(spikes, axis=1) / delta_t
    return mean_spikes

def get_orientation(idx_list, N):
    preferred_directions = 180 * (idx_list*2+1) / N # ?? not sure
    
    return preferred_directions

def get_theta_time_series(spikes, N = ct.N):

    theta_time_series = []
    
    for t in range(spikes.shape[0]):
        idx_list = np.arange(N) # not sure what idx_list is supposed to be. Took all neurons but supposed to take only the ones that spiked?
        
        # Calculate the preferred direction for each neuron index
        preferred_directions = get_orientation(idx_list, N)
        
        # Calculate the spike count for each neuron index at time t
        spike_counts = spikes[t, idx_list]
        
        # Calculate the population vector
        population_vector = np.mean(preferred_directions[spike_counts != 0]/(spike_counts[spike_counts != 0]))

        theta_time_series.append(population_vector)
    
    return np.array(theta_time_series)

def get_bump(spikes, N=ct.N):
    """
    Calculate the bump location for each time step based on the given spike data.

    Parameters:
    spikes (ndarray): Array of spike data.
    N (int): Number of neurons.

    Returns:
    list: List of population bump location for each time step.
    """

    x = np.linspace(0, 2*np.pi, N)
    theta_time_series = []

    for t in range(spikes.shape[0]):
        angle = x[spikes[t] != 0]

        # Calculate the population vector
        population_vector = np.arctan2(np.sum(np.sin(angle)), np.sum(np.cos(angle)))
        if population_vector < 0:
            population_vector += 2*np.pi
        theta_time_series.append(population_vector)

    return theta_time_series

def smooth_random_trajectory(total_time, time_step, speed, max_delta_theta):
    num_steps = int(total_time / time_step)
    

    positions = np.zeros((num_steps, 2))
    head_directions = np.zeros(num_steps)
    positions[0] = np.random.rand(2)  # Random initial position in [0, 1] x [0, 1]
    head_directions[0] = np.random.uniform(0, 2*np.pi)  # Random initial head direction
    
    # Generate trajectory
    for t in range(1, num_steps):

        # Generate random step in polar coordinates
        theta = np.random.uniform(-max_delta_theta, max_delta_theta)  # Random direction within a narrow range
        r = speed * time_step  # Magnitude of step

        head_directions[t] = head_directions[t-1] + theta
        # not needed because of cos invariance and better for angle smoothness
        # head_directions[t] = head_directions[t] % (2 * np.pi) 
        
        # Convert polar step to Cartesian step
        dx = r * np.cos(head_directions[t])
        dy = r * np.sin(head_directions[t])
        
        # Update position
        positions[t] = positions[t-1] + np.array([dx, dy])
    
    return head_directions, positions


