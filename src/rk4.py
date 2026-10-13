"""
RK4 Solver for Lorenz System
Implements 4th-order Runge-Kutta integration for the Lorenz equations.
"""

import numpy as np


class LorenzRK4:
    """
    Lorenz system solver using 4th-order Runge-Kutta method.
    
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
        Perform one RK4 step.
        
        Args:
            state: Current state [x, y, z]
            dt: Time step
            
        Returns:
            New state after time step dt
        """
        k1 = self._lorenz_derivative(state)
        k2 = self._lorenz_derivative(state + 0.5 * dt * k1)
        k3 = self._lorenz_derivative(state + 0.5 * dt * k2)
        k4 = self._lorenz_derivative(state + dt * k3)
        
        return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    def solve(self, initial_state, dt=0.01, T=25.0, checkpoint=20):
        """
        Solve the Lorenz system over time interval [0, T].
        
        Args:
            initial_state: Initial condition [x0, y0, z0]
            dt: Time step (default: 0.01)
            T: Total simulation time (default: 25.0)
            checkpoint: Store trajectory every checkpoint steps (default: 20)
            
        Returns:
            times: Array of time points
            trajectory: Array of states at each time point
        """
        num_steps = int(T / dt)
        times = []
        trajectory = [initial_state.copy()]
        
        state = initial_state.copy()
        
        for i in range(num_steps):
            state = self.step(state, dt)
            
            if (i + 1) % checkpoint == 0:
                times.append((i + 1) * dt)
                trajectory.append(state.copy())
        
        return np.array(times), np.array(trajectory)
    
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


def generate_trajectories(num_trajectories=100, dt=0.01, T=25.0, 
                         sigma=10.0, rho=28.0, beta=8.0/3.0):
    """
    Generate multiple Lorenz trajectories with random initial conditions.
    
    Args:
        num_trajectories: Number of trajectories to generate
        dt: Time step
        T: Total simulation time
        sigma, rho, beta: Lorenz parameters
        
    Returns:
        trajectories: List of full trajectories
        times: Time grid
    """
    solver = LorenzRK4(sigma=sigma, rho=rho, beta=beta)
    trajectories = []
    
    # Generate initial conditions near the attractor
    np.random.seed(42)
    for i in range(num_trajectories):
        x0 = np.random.uniform(-15, 15)
        y0 = np.random.uniform(-15, 25)
        z0 = np.random.uniform(5, 35)
        initial_state = np.array([x0, y0, z0])
        
        times, trajectory = solver.solve_full(initial_state, dt=dt, T=T)
        trajectories.append(trajectory)
    
    return trajectories, times
