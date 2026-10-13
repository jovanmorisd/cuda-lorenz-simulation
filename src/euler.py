"""
Euler Solver for Lorenz System
Implements simple Euler integration for the Lorenz equations.
"""

import numpy as np


class LorenzEuler:
    """
    Lorenz system solver using simple Euler integration method.
    
    The Lorenz equations are:
    dx/dt = sigma * (y - x)
    dy/dt = x * (rho - z) - y
    dz/dt = x * y - beta * z
    """
    
    def __init__(self, sigma=10.0, rho=28.0, beta=8.0/3.0):
        """
        Initialize Lorenz system parameters.
        
        Args:
            sigma: Parameter sigma (default: 10.0)
            rho: Parameter rho (default: 28.0)
            beta: Parameter beta (default: 8/3)
        """
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
    
    def _lorenz_derivative(self, state):
        """
        Compute the derivative of the Lorenz system.
        
        Args:
            state: [x, y, z]
            
        Returns:
            [dx/dt, dy/dt, dz/dt]
        """
        x, y, z = state
        dx_dt = self.sigma * (y - x)
        dy_dt = x * (self.rho - z) - y
        dz_dt = x * y - self.beta * z
        return np.array([dx_dt, dy_dt, dz_dt])
    
    def step(self, state, dt):
        """
        Perform one Euler step.
        
        Args:
            state: Current state [x, y, z]
            dt: Time step
            
        Returns:
            New state after time step dt
        """
        derivative = self._lorenz_derivative(state)
        return state + dt * derivative
    
    def solve_full(self, initial_state, dt=0.01, T=25.0):
        """
        Solve the Lorenz system and return all time steps.
        
        Args:
            initial_state: Initial condition [x0, y0, z0]
            dt: Time step (default: 0.01)
            T: Total simulation time (default: 25.0)
            
        Returns:
            times: Array of all time points
            trajectory: Array of all states
        """
        num_steps = int(T / dt)
        times = np.arange(0, T, dt)[:num_steps]
        trajectory = np.zeros((num_steps, 3))
        
        trajectory[0] = initial_state.copy()
        state = initial_state.copy()
        
        for i in range(1, num_steps):
            state = self.step(state, dt)
            trajectory[i] = state.copy()
        
        return times, trajectory
