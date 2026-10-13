/* CUDA Lorenz System Solver
 * Each thread computes one complete trajectory
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <chrono>
#include <fstream>

// Lorenz parameters
#define SIGMA 10.0f
#define RHO 28.0f
#define BETA (8.0f / 3.0f)

// Simulation parameters
#define DT 0.01f
#define T_MAX 25.0f
#define NUM_STEPS 2500  // T / dt

// Trajectory output stride
#define STRIDE 1

/**
 * CUDA kernel for Lorenz simulation
 * Each block computes multiple trajectories, each thread in block computes one trajectory
 */
__global__ void lorenzKernel(
    float *trajectories,      // Output: trajectories[num_trajectories][NUM_STEPS][3]
    float *initial_states,    // Input: initial_states[num_trajectories][3]
    int num_trajectories
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= num_trajectories) return;
    
    // Load initial state
    float x = initial_states[tid * 3 + 0];
    float y = initial_states[tid * 3 + 1];
    float z = initial_states[tid * 3 + 2];
    
    // Store initial state
    int base_idx = tid * NUM_STEPS * 3;
    trajectories[base_idx + 0] = x;
    trajectories[base_idx + 1] = y;
    trajectories[base_idx + 2] = z;
    
    // Integrate using RK4
    for (int step = 1; step < NUM_STEPS; step++) {
        // k1
        float k1x = SIGMA * (y - x);
        float k1y = x * (RHO - z) - y;
        float k1z = x * y - BETA * z;
        
        // k2
        float x2 = x + 0.5f * DT * k1x;
        float y2 = y + 0.5f * DT * k1y;
        float z2 = z + 0.5f * DT * k1z;
        float k2x = SIGMA * (y2 - x2);
        float k2y = x2 * (RHO - z2) - y2;
        float k2z = x2 * y2 - BETA * z2;
        
        // k3
        float x3 = x + 0.5f * DT * k2x;
        float y3 = y + 0.5f * DT * k2y;
        float z3 = z + 0.5f * DT * k2z;
        float k3x = SIGMA * (y3 - x3);
        float k3y = x3 * (RHO - z3) - y3;
        float k3z = x3 * y3 - BETA * z3;
        
        // k4
        float x4 = x + DT * k3x;
        float y4 = y + DT * k3y;
        float z4 = z + DT * k3z;
        float k4x = SIGMA * (y4 - x4);
        float k4y = x4 * (RHO - z4) - y4;
        float k4z = x4 * y4 - BETA * z4;
        
        // Combine
        x = x + (DT / 6.0f) * (k1x + 2*k2x + 2*k3x + k4x);
        y = y + (DT / 6.0f) * (k1y + 2*k2y + 2*k3y + k4y);
        z = z + (DT / 6.0f) * (k1z + 2*k2z + 2*k3z + k4z);
        
        // Store state
        int idx = base_idx + step * 3;
        trajectories[idx + 0] = x;
        trajectories[idx + 1] = y;
        trajectories[idx + 2] = z;
    }
}

/**
 * Initialize random states on GPU
 */
__global__ void initializeStates(
    float *initial_states,
    int num_trajectories,
    float seed
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= num_trajectories) return;
    
    // Simple deterministic initialization (for reproducibility)
    // Better: use curand
    float x = -15.0f + (tid % 100) * 0.6f;
    float y = -15.0f + ((tid / 10) % 40) * 1.0f;
    float z = 5.0f + ((tid / 400) % 30) * 1.0f;
    
    initial_states[tid * 3 + 0] = x;
    initial_states[tid * 3 + 1] = y;
    initial_states[tid * 3 + 2] = z;
}

int main(int argc, char *argv[]) {
    int num_trajectories = 100;
    if (argc > 1) {
        num_trajectories = atoi(argv[1]);
    }
    
    // Allocate device memory
    float *d_trajectories, *d_initial_states;
    size_t traj_size = num_trajectories * NUM_STEPS * 3 * sizeof(float);
    size_t state_size = num_trajectories * 3 * sizeof(float);
    
    cudaMalloc(&d_trajectories, traj_size);
    cudaMalloc(&d_initial_states, state_size);
    
    // Initialize states
    int block_size = 256;
    int num_blocks = (num_trajectories + block_size - 1) / block_size;
    
    initializeStates<<<num_blocks, block_size>>>(d_initial_states, num_trajectories, 42.0f);
    cudaDeviceSynchronize();
    
    // Start timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    // Launch kernel
    lorenzKernel<<<num_blocks, block_size>>>(d_trajectories, d_initial_states, num_trajectories);
    cudaDeviceSynchronize();
    
    // End timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed = 0;
    cudaEventElapsedTime(&elapsed, start, stop);
    
    // Copy results back to host
    float *h_trajectories = (float *)malloc(traj_size);
    cudaMemcpy(h_trajectories, d_trajectories, traj_size, cudaMemcpyDeviceToHost);
    
    // Output results
    printf("CUDA Lorenz Simulation Results\n");
    printf("==============================\n");
    printf("Number of trajectories: %d\n", num_trajectories);
    printf("Simulation time (T): %.1f\n", T_MAX);
    printf("Time step (dt): %.4f\n", DT);
    printf("Number of steps: %d\n", NUM_STEPS);
    printf("Execution time: %.2f ms\n", elapsed);
    printf("Time per trajectory: %.4f ms\n", elapsed / num_trajectories);
    
    // Save trajectories
    FILE *outfile = fopen("cuda_trajectories.txt", "w");
    for (int i = 0; i < num_trajectories; i++) {
        for (int j = 0; j < NUM_STEPS; j++) {
            int idx = i * NUM_STEPS * 3 + j * 3;
            fprintf(outfile, "%f %f %f\n", 
                   h_trajectories[idx], 
                   h_trajectories[idx+1], 
                   h_trajectories[idx+2]);
        }
        fprintf(outfile, "\n");
    }
    fclose(outfile);
    
    printf("Trajectories saved to cuda_trajectories.txt\n");
    printf("Execution time: %.2f ms\n", elapsed);
    
    // Cleanup
    cudaFree(d_trajectories);
    cudaFree(d_initial_states);
    free(h_trajectories);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
