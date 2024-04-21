import numpy as np
from tqdm import tqdm
import constant as ct

def g(h, alpha=ct.alpha, beta=ct.beta):
    '''This function calculates the value of the transfer function g(h) of Poisson neurons given the values of h, alpha, and beta.
    
    Parameters:
    h (np.ndarray): The potential of every neuron
    alpha (float): The value of alpha used in the transfer function.
    beta (float): The value of beta used in the transfer function.
    '''

    return 1/(1 + np.exp(-2 * alpha * (h - beta)))

def oscillating_input(t, omega=ct.omega, I_0=ct.I_0):
    """
    Compute the input current for the neurons.

    Parameters:
    - t (ndarray): time points.
    - omega (float): Frequency of the input current.
    - I_0 (float): Amplitude of the input current.

    Returns:
    - I (ndarray): input current for the neurons.
    """
    I = I_0 * np.sin(omega * t)

    return I

def external_input(t,x, mu1=ct.mu1, mu2=ct.mu2, sigma=ct.sigma):
    """
    Compute the input current for the neurons.

    Parameters:
    - t (ndarray): time points.
    - x (ndarray): Neuron positions.
    - mu1 (float): Mean of the first Gaussian.
    - mu2 (float): Mean of the second Gaussian.
    - sigma (float): Standard deviation of the Gaussians.

    Returns:
    - I (ndarray): input current for the neurons.
    """
    if 300<=t<400 :
        I  = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-((x - mu1) ** 2 / (2*sigma**2)))
        
    elif 600<=t<700 :
        I  = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-((x - mu2) ** 2 / (2*sigma**2)))
    else :
        I = 0
    return I

def recurrent_interactions_input(x, S, J=ct.J):
    """
    Compute the input current for the neurons based on recurrent interactions.

    Parameters:
    - x (ndarray): Neuron positions.
    - S (ndarray): Spike trains of the neurons.
    - J (float): Interaction strength.

    Returns:
    - I (ndarray): Input current for the neurons.
    """
    mc = np.mean(np.cos(x)*S)
    ms = 1/len(x)*(np.sin(x)@S)
    I = J * (np.cos(x) * mc + np.sin(x) * ms)
    
    return I

def recurrent_interactions_input(x, S, J=ct.J, phi = 0):
    """
    Compute the input current for the neurons based on recurrent interactions with a small angle phi.

    Parameters:
    - x (ndarray): Neuron positions.
    - S (ndarray): Spike trains of the neurons.
    - J (float): Interaction strength.

    Returns:
    - I (ndarray): Input current for the neurons.
    """
    mc = np.mean(np.cos(x)*S)
    ms = np.mean(np.sin(x)*S)
    I = J * (np.cos(x-phi) * mc + np.sin(x-phi) * ms)
    print(I)
    
    return I

def line_input(x, S, J, J0, J1, sigma_w):

    sum_J0 =J0*np.sum(S)
    gaussian = J1*np.sum(np.exp((-(x-x.T)**2)/(2*sigma_w**2)), axis = 0)
    I = J/len(x) * (sum_J0 + gaussian)
    # print(I)
    return I


def spike_simulation(input_fct, initial_voltage, N=ct.N, delta_t=ct.delta_t, tau=ct.tau, T=ct.T, R=ct.R, r_0=ct.r_0, alpha=ct.alpha, beta=ct.beta, J0= ct.J0, J1 = ct.J1, sigma_w = ct.sigma_w, phi = 0,  J=ct.J, omega=None, I_0=None, theory = False, I_ext=False) : 
    """
    Simulates spike generation in a population of neurons.

    Parameters:
    - N (int): Number of neurons in the population.
    - delta_t (float): Time step size for simulation.
    - tau (float): Membrane time constant.
    - T (float): Total simulation time.
    - R (float): Resistance of the neuron.

    Returns:
    - h (ndarray): membrane potential of each neuron over time.
    - mean_spikes (ndarray): mean spike occurence over 1 ms bins.
    """
    h = np.zeros((int(T/delta_t), N)) # e.g. (10000, 100)
    r = np.zeros((int(T/delta_t), N))
    s = np.zeros((int(T/delta_t), N))
    
    # Initialize the position of each neuron
    x = np.linspace(0, 2*np.pi, N)
    
    h[0, :] = initial_voltage 
    r[0, :] = r_0 * g(h[0, :], alpha, beta)
    s[0, :] =  np.random.binomial(1,r[0] * delta_t)

    for t in tqdm(range(h.shape[0]-1)):
        
        #compute current
        if input_fct == oscillating_input:
            I = input_fct(t*delta_t, omega, I_0)
        elif input_fct == recurrent_interactions_input:
            I = input_fct(x, s[t]/delta_t, J, phi)
        elif input_fct == line_input:
            I = input_fct(x.reshape(-1,1), s[t]/delta_t, J, J0, J1, sigma_w)   
    
        else:
            ValueError("Input function not recognized.")
            
        if I_ext:
            I += external_input(t*delta_t, x)
            
        h[t+1] = h[t] + delta_t/tau * (-h[t] + R * I)
        r[t+1] = r_0 * g(h[t+1], alpha, beta)
        s[t+1] = np.random.binomial(1, r[t+1] * delta_t)
        
    if theory :
        print("theory")
        return h, r*delta_t
    else :
        return h, s
    
def bins_spike (spikes, bins, delta_t= ct.delta_t, N=ct.N, mean = False):
    bins_size = int(bins/delta_t)

    spikes = spikes.reshape((int(spikes.shape[0]//bins_size), bins_size, N))
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
