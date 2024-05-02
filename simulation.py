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

class PoissonNeuron:
    def __init__(self, N=ct.N, delta_t=ct.delta_t, tau=ct.tau, 
                 T=ct.T, R=ct.R, r_0=ct.r_0, alpha=ct.alpha, beta=ct.beta, 
                 J0=ct.J0, J1=ct.J1, sigma_w=ct.sigma_w, phi=0, J=ct.J, 
                 omega=ct.omega, I_0=ct.I_0, mu1=ct.mu1, mu2=ct.mu2, 
                 sigma=ct.sigma, I_ext=False, theta_H=None):
        self.N = N # Number of neurons
        self.delta_t = delta_t # Time step size for simulation
        self.tau = tau # Membrane time constant
        self.T = T # Total simulation time
        self.R = R # Resistance of the neuron
        self.r_0 = r_0 # Sigmoid normalisation factor
        self.alpha = alpha # Sigmoid parameter 1
        self.beta = beta # Sigmoid parameter 2
        self.J0 = J0 # Base interaction strength
        self.J1 = J1 # Gaussian interaction term
        self.sigma_w = sigma_w # Standard deviation of the Gaussian interaction
        self.phi = phi # Position phase difference
        self.J = J # Interaction strength
        self.omega = omega # Frequency of the input current
        self.I_0 = I_0 # Amplitude of the input current
        self.mu1=mu1 # External input parameter 1
        self.mu2=mu2 # External input parameter 1
        self.sigma=sigma # External input parameter 1
        self.I_ext = I_ext # External input flag
        self.theta_H = theta_H # Head directions, same dims as x

    def oscillating_input(self, t):
        """
        Compute the input current for the neurons.

        Parameters:
        - t (ndarray): time points.
        - omega (float): Frequency of the input current.
        - I_0 (float): Amplitude of the input current.

        Returns:
        - I (ndarray): input current for the neurons.
        """
        I = self.I_0 * np.sin(self.omega * t)

        return I

    def external_input(self, t):
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
            I  = 1/(self.sigma * np.sqrt(2 * np.pi)) * np.exp(-((self.x - self.mu1) ** 2 / (2*self.sigma**2)))
            
        elif 600<=t<700 :
            I  = 1/(self.sigma * np.sqrt(2 * np.pi)) * np.exp(-((self.x - self.mu2) ** 2 / (2*self.sigma**2)))
        else :
            I = 0
        return I

    def recurrent_interactions_input(self, S):
        """
        Compute the input current for the neurons based on recurrent interactions with a small angle phi.

        Parameters:
        - x (ndarray): Neuron positions.
        - S (ndarray): Spike trains of the neurons.
        - J (float): Interaction strength.

        Returns:
        - I (ndarray): Input current for the neurons.
        """
        mc = np.mean(np.cos(self.x)*S)
        ms = np.mean(np.sin(self.x)*S)
        I = self.J * (np.cos(self.x-self.phi) * mc + np.sin(self.x-self.phi) * ms)

        return I

    def line_input(self, S):

        I = self.J / self.N * self.gaussian @ S 

        return I
    
    def head_external_input(self, theta_H):

        return self.I_0 * np.cos(self.x-theta_H)

    def spike_simulation(self, input_fct, initial_voltage, theory = False) : 
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
        self.h = np.zeros((int(self.T / self.delta_t), self.N))
        self.r = np.zeros((int(self.T / self.delta_t), self.N))
        self.s = np.zeros((int(self.T / self.delta_t), self.N))

        self.x = np.linspace(0, 2*np.pi, self.N)
        if input_fct == self.line_input:
            x_reshaped = self.x.reshape(-1, 1)
            self.gaussian = self.J0 + self.J1 * np.exp((-(x_reshaped-self.phi-x_reshaped.T)**2)/(2*self.sigma_w**2))

        self.h[0, :] = initial_voltage
        self.r[0, :] = self.r_0 * g(self.h[0, :])
        self.s[0, :] = np.random.binomial(1, self.r[0] * self.delta_t)


        for t in tqdm(range(self.h.shape[0]-1)):
            
            #compute current
            if input_fct == self.oscillating_input:
                I = input_fct(t*self.delta_t)
            elif input_fct == self.recurrent_interactions_input:
                I = input_fct(self.s[t]/self.delta_t)
            elif input_fct == self.line_input:
                I = input_fct(self.s[t]/self.delta_t)   
            elif input_fct == self.head_external_input:
                I = input_fct(self.theta_H[t])
        
            else:
                ValueError("Input function not recognized.")
                
            if self.I_ext:
                I += self.external_input(t*self.delta_t)
                
            self.h[t+1] = self.h[t] + self.delta_t/self.tau * (-self.h[t] + self.R * I)
            self.r[t+1] = self.r_0 * g(self.h[t+1], self.alpha, self.beta)
            self.s[t+1] = np.random.binomial(1, self.r[t+1] * self.delta_t)
            
        if theory :
            print("theory")
            return self.h, self.r*self.delta_t
        else :
            return self.h, self.s

class TwoPopulationSimulation:
    def __init__(self, N=ct.N, delta_t=ct.delta_t, tau=ct.tau, T=ct.T, R=ct.R, r_0=ct.r_0, alpha=ct.alpha, beta=ct.beta, J=ct.J, theta=ct.theta, I0 = 0, I_ext=False):
        self.N = N
        self.delta_t = delta_t
        self.tau = tau
        self.T = T
        self.R = R
        self.r_0 = r_0
        self.alpha = alpha
        self.beta = beta
        self.J = J
        self.theta = theta
        self.I_ext = I_ext
        self.I0 = I0
        
    def initialize_simulation(self):
        h = np.zeros((int(self.T/self.delta_t), self.N))
        r = np.zeros((int(self.T/self.delta_t), self.N))
        s = np.zeros((int(self.T/self.delta_t), self.N))
        
        return h, r, s
    
    def initialize_positions(self):
        x = np.linspace(0, 2*np.pi, self.N)
        return x
    
    def uniform_voltage(self):
        return np.random.uniform(0, 1, self.N) 
    
    def centered_voltage(self):
        
        initial_voltage = np.linspace(0, 1, self.N)
        initial_voltage = 1 / (0.25*np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((initial_voltage - 0.5) / 0.25) ** 2)
        initial_voltage = initial_voltage / np.max(initial_voltage)
        initial_voltage += np.random.uniform(-0.1, 0.1, self.N)

        return initial_voltage

    
    def first_step_update(self, initial_voltage):
        h0 =  initial_voltage()
        r0 = self.r_0 * g(h0, self.alpha, self.beta)
        s0 =  np.random.binomial(1,r0 * self.delta_t)
        
        return h0, r0, s0
    
    def update(self, t, IL, IR):

        self.hL[t+1] = self.hL[t] + self.delta_t/self.tau * (-self.hL[t] + self.R * IL)
        self.rL[t+1] = self.r_0 * g(self.hL[t+1], self.alpha, self.beta)
        self.sL[t+1] = np.random.binomial(1, self.rL[t+1] * self.delta_t)
        
        self.hR[t+1] = self.hR[t] + self.delta_t/self.tau * (-self.hR[t] + self.R * IR)
        self.rR[t+1] = self.r_0 * g(self.hR[t+1], self.alpha, self.beta)
        self.sR[t+1] = np.random.binomial(1, self.rR[t+1] * self.delta_t)
    
    def two_population_recurrent_input(self, xL, xR, SL, SR, J=ct.J, theta = 0):

        mcos_L = np.mean(np.cos(xL)*SL)
        mcos_R = np.mean(np.cos(xR)*SR)
        msin_L = np.mean(np.sin(xL)*SL)
        msin_R = np.mean(np.sin(xR)*SR)
        IL = J * (np.cos(xL+theta) * (mcos_L + mcos_R) + np.sin(xL+theta) * (msin_L + msin_R))
        IR = J * (np.cos(xR-theta) * (mcos_L + mcos_R) + np.sin(xR-theta) * (msin_L + msin_R))
        
        return IL, IR
    
    def external_input(self, t):
        if 300<=t<600 :
            I  = self.I0
        else :
            I = 0
        return I
        
    def simulation(self, initial_voltage = uniform_voltage) : 
    
        self.hL, self.rL, self.sL = self.initialize_simulation()
        self.hR, self.rR, self.sR = self.initialize_simulation()

        # Initialize the position of each neuron
        self.xL = self.initialize_positions()
        self.xR = self.initialize_positions()
        
        self.hL[0, :], self.rL[0, :], self.sL[0, :] =  self.first_step_update(initial_voltage)
        self.hR[0, :], self.rR[0, :], self.sR[0, :] =  self.first_step_update(initial_voltage)

        for t in tqdm(range(self.hR.shape[0]-1)):
            
            IL, IR = self.two_population_recurrent_input(self.xL, self.xR, self.sL[t]/self.delta_t, self.sR[t]/self.delta_t,  self.J, self.theta)
            
            if self.I_ext:
                IL -= self.external_input(t*self.delta_t) 
                IR += self.external_input(t*self.delta_t)  
                
            self.update(t, IL, IR)

        return self.hL, self.sL, self.hR, self.sR
    

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


