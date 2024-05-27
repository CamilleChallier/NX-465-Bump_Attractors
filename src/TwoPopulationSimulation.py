import numpy as np
from tqdm import tqdm
import constant as ct
from simulation import g

class TwoPopulationSimulation:
    def __init__(self, N=ct.N, delta_t=ct.delta_t, tau=ct.tau, 
                 T=ct.T, R=ct.R, r_0=ct.r_0, alpha=ct.alpha, beta=ct.beta, 
                 J=ct.J, theta=ct.theta, I0 = 0, I_ext=False, 
                 J_head=ct.J_head, head_population=None, x=True):
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
        self.J_head = J_head
        if head_population is not None and x==True: 
            self.w_head = np.cos(head_population.x)
            self.I_head = self.J_head * np.mean(head_population.s/self.delta_t * self.w_head, axis =1)
        elif head_population is not None and x==False: 
            self.w_head = np.sin(head_population.x)
            self.I_head = self.J_head * np.mean(head_population.s/self.delta_t * self.w_head, axis =1)
        else: self.w_head = None
        
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
            # print(np.mean(IL), np.mean(IR))
            if self.I_ext:
                IL -= self.external_input(t*self.delta_t) 
                IR += self.external_input(t*self.delta_t)  

            if self.w_head is not None:
                # print(np.mean(self.I_head[t]))
                IL -= self.I_head[t]
                IR += self.I_head[t]

            self.update(t, IL, IR)
        return self.hL, self.sL, self.hR, self.sR
    
