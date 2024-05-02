import numpy as np
from tqdm import tqdm
import constant as ct
from simulation import g

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
        self.theta_H = theta_H # Head directions, dim = T/delta_t

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