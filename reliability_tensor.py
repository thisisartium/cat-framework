import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plot_bars = True

# Set random seed for reproducibility
np.random.seed(3)

# Parameters
N = 24  # Number of inputs
M = 24  # Number of outputs per input
K = 24  # Number of validators

# Success probabilities for validators (hovering between 0.90 and 1.0)
p_k = np.random.uniform(low=0.95, high=1.0, size=K)

num_problematic = max(1, K // 10)  # Ensure at least 1 element is problematic

# Randomly select indices for the problematic values
problematic_indices = np.random.choice(K, num_problematic, replace=False)

# Set the selected indices to 0.5
p_k[problematic_indices] = 0.5

# Initialize IRF array with ones
IRF = np.ones(N)

# Calculate 10% of N
num_problematic = max(1, N // 10)  # Ensure at least 1 element is problematic

# Randomly select indices for the problematic values
problematic_indices = np.random.choice(N, num_problematic, replace=False)

# Set the selected indices to 0.5
IRF[problematic_indices] = 0.2

# Initialize Reliability Tensor R[N][M][K]
R = np.zeros((N, M, K), dtype=int)

# Simulate pass/fail status
for i in range(N):
    for j in range(M):
        for k in range(K):
            p_ik = p_k[k] * IRF[i]
            r = np.random.rand()
            if r < p_ik:
                R[i][j][k] = 1  # Pass
            else:
                R[i][j][k] = 0  # Fail

# Compute Input Marginal Success Percentage (Input MSP)
Input_MSP = np.sum(R, axis=(1,2)) / (M * K)

# Compute Validator Marginal Success Percentage (Validator MSP)
Validator_MSP = np.sum(R, axis=(0,1)) / (N * M)

# Compute Overall Success Percentage
Overall_Success_Percentage = np.sum(R) / (N * M * K)

# Print Overall Success Percentage
print(f"Overall Success Percentage: {Overall_Success_Percentage:.2%}")

# Print Input MSP
print("\nInput Marginal Success Percentages:")
for i in range(N):
    print(f"Input {i+1}: {Input_MSP[i]:.2%}")

# Print Validator MSP
print("\nValidator Marginal Success Percentages:")
for k in range(K):
    print(f"Validator {k+1}: {Validator_MSP[k]:.2%}")

if plot_bars:
    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot Input MSP on the first subplot
    axes[0].bar(range(1, N + 1), Input_MSP, color='skyblue')
    axes[0].set_xlabel('Input Index')
    axes[0].set_ylabel('Success Percentage')
    axes[0].set_title('Input Marginal Success Percentage')
    axes[0].set_ylim(0, 1)
    axes[0].grid(axis='y')

    # Plot Validator MSP on the second subplot
    axes[1].bar(range(1, K + 1), Validator_MSP, color='salmon')
    axes[1].set_xlabel('Validator Index')
    axes[1].set_ylabel('Success Percentage')
    axes[1].set_title('Validator Marginal Success Percentage')
    axes[1].set_ylim(0, 1)
    axes[1].grid(axis='y')

    # Adjust the layout for better spacing
    plt.tight_layout()

    # Show the plot
    plt.show()

# Visualize Reliability Tensor
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Prepare the voxel positions
x, y, z = np.indices((N+1, M+1, K+1))
x = x[:-1, :-1, :-1].flatten()
y = y[:-1, :-1, :-1].flatten()
z = z[:-1, :-1, :-1].flatten()

# Prepare the colors based on pass/fail
colors = np.empty((N, M, K), dtype=object)
for i in range(N):
    for j in range(M):
        for k in range(K):
            if R[i, j, k] == 1:
                colors[i, j, k] = '#00FF00'  # Green for pass
            else:
                colors[i, j, k] = '#FF0000'  # Red for fail

# Flatten colors array
colors = colors.flatten()

# Plot the voxels
ax.bar3d(x, y, z, 1, 1, 1, color=colors, edgecolor='black', shade=True)

# Set labels and title
ax.set_xlabel('Input Index')
ax.set_ylabel('Output Index')
ax.set_zlabel('Validator Index')
ax.set_title('Reliability Tensor')

# Adjust ticks
ax.set_xticks(np.arange(0.5, N+0.5, 1))
ax.set_yticks(np.arange(0.5, M+0.5, 1))
ax.set_zticks(np.arange(0.5, K+0.5, 1))
ax.set_xticklabels(range(1, N+1))
ax.set_yticklabels(range(1, M+1))
ax.set_zticklabels(range(1, K+1))

plt.show()
