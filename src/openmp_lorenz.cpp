#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <omp.h>
#include <cmath>
#include <fstream>

// Lorenz parameters
const float sigma = 10.0f;
const float rho = 28.0f;
const float beta = 8.0f / 3.0f;

// Simulation parameters
const float dt = 0.01f;
const float T = 25.0f;
const int checkpoint = 20;

// Structure to hold a 3D state
struct State {
    float x, y, z;
};

/**
 * Compute Lorenz derivative
 */
void lorenzDerivative(const State& state, State& deriv) {
    deriv.x = sigma * (state.y - state.x);
    deriv.y = state.x * (rho - state.z) - state.y;
    deriv.z = state.x * state.y - beta * state.z;
}

/**
 * Perform one RK4 step
 */
void rk4Step(const State& current, State& next, float dt) {
    State k1, k2, k3, k4;
    State temp;
    
    // k1
    lorenzDerivative(current, k1);
    
    // k2
    temp.x = current.x + 0.5f * dt * k1.x;
    temp.y = current.y + 0.5f * dt * k1.y;
    temp.z = current.z + 0.5f * dt * k1.z;
    lorenzDerivative(temp, k2);
    
    // k3
    temp.x = current.x + 0.5f * dt * k2.x;
    temp.y = current.y + 0.5f * dt * k2.y;
    temp.z = current.z + 0.5f * dt * k2.z;
    lorenzDerivative(temp, k3);
    
    // k4
    temp.x = current.x + dt * k3.x;
    temp.y = current.y + dt * k3.y;
    temp.z = current.z + dt * k3.z;
    lorenzDerivative(temp, k4);
    
    // Combine
    next.x = current.x + (dt / 6.0f) * (k1.x + 2*k2.x + 2*k3.x + k4.x);
    next.y = current.y + (dt / 6.0f) * (k1.y + 2*k2.y + 2*k3.y + k4.y);
    next.z = current.z + (dt / 6.0f) * (k1.z + 2*k2.z + 2*k3.z + k4.z);
}

/**
 * Solve single trajectory
 */
void solveSingleTrajectory(const State& initial, std::vector<State>& trajectory) {
    int num_steps = static_cast<int>(T / dt);
    trajectory.resize(num_steps);
    
    State current = initial;
    trajectory[0] = current;
    
    for (int i = 1; i < num_steps; i++) {
        rk4Step(current, current, dt);
        trajectory[i] = current;
    }
}

int main(int argc, char* argv[]) {
    int num_trajectories = 100;
    if (argc > 1) {
        num_trajectories = std::atoi(argv[1]);
    }
    
    // Random number generation
    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis_x(-15, 15);
    std::uniform_real_distribution<> dis_y(-15, 25);
    std::uniform_real_distribution<> dis_z(5, 35);
    
    // Generate initial conditions
    std::vector<State> initial_states(num_trajectories);
    for (int i = 0; i < num_trajectories; i++) {
        initial_states[i].x = dis_x(gen);
        initial_states[i].y = dis_y(gen);
        initial_states[i].z = dis_z(gen);
    }
    
    // Storage for trajectories
    std::vector<std::vector<State>> trajectories(num_trajectories);
    
    // Start timing
    auto start = std::chrono::high_resolution_clock::now();
    
    // Parallel region - each thread computes one trajectory
    #pragma omp parallel for
    for (int i = 0; i < num_trajectories; i++) {
        solveSingleTrajectory(initial_states[i], trajectories[i]);
    }
    
    // End timing
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Output results
    std::cout << "OpenMP Lorenz Simulation Results" << std::endl;
    std::cout << "================================" << std::endl;
    std::cout << "Number of trajectories: " << num_trajectories << std::endl;
    std::cout << "Simulation time (T): " << T << std::endl;
    std::cout << "Time step (dt): " << dt << std::endl;
    std::cout << "Number of steps per trajectory: " << static_cast<int>(T / dt) << std::endl;
    std::cout << "Execution time: " << duration.count() << " ms" << std::endl;
    std::cout << "Time per trajectory: " << (double)duration.count() / num_trajectories << " ms" << std::endl;
    
    // Save trajectories to file
    std::ofstream outfile("openmp_trajectories.txt");
    int num_steps = trajectories[0].size();
    
    for (int i = 0; i < num_trajectories; i++) {
        for (int j = 0; j < num_steps; j++) {
            outfile << trajectories[i][j].x << " "
                   << trajectories[i][j].y << " "
                   << trajectories[i][j].z << "\n";
        }
        outfile << "\n";
    }
    outfile.close();
    
    std::cout << "Trajectories saved to openmp_trajectories.txt" << std::endl;
    std::cout << "Execution time: " << duration.count() << std::endl;
    
    return 0;
}
