import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

def plot_policy_heatmap(pi, save_path=None):
    # Create a grid representing states
    states = np.zeros((21, 21))
    
    # Fill the grid with corresponding actions
    for i in range(21):
        for j in range(21):
            states[i, j] = pi[(i, j)]
    
    # Normalize the colormap
    norm = Normalize(vmin=-5, vmax=5)
    
    # Plot the heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    heatmap = ax.imshow(states, cmap='coolwarm', norm=norm, origin='lower')
    
    # Add text annotations within each square of the heatmap
    for y in range(states.shape[0]):
        for x in range(states.shape[1]):
            plt.text(x, y, int(states[y, x]), ha='center', va='center', color='white', fontsize=8)
    
    # Set the ticks to the center of each cell
    ax.set_xticks(np.arange(states.shape[1]))
    ax.set_yticks(np.arange(states.shape[0]))
    
    # Set the tick labels
    ax.set_xticklabels(np.arange(states.shape[1]))
    ax.set_yticklabels(np.arange(states.shape[0]))
    
    # Turn off the grid if you don't want it to interfere with the cell borders
    ax.grid(False)
    
    # Set the aspect of the plot to 'equal' to ensure squares are displayed as squares
    ax.set_aspect('equal')
    
    # Labeling
    plt.colorbar(heatmap, ax=ax, label='Action')
    plt.xlabel('Cars at Location B')
    plt.ylabel('Cars at Location A')
    plt.title('Policy Heatmap')
    
    # Adjust the limits to ensure squares are displayed as squares
    ax.set_xlim(-0.5, 20.5)
    ax.set_ylim(-0.5, 20.5)
    
    # Save the figure if save_path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()

def main():
    # Load the final policy
    with open('pi0', 'rb') as f:
        pi = pickle.load(f)
    
    # Plot the heatmap and save the result
    save_path = '1times_iteration_origin.png'
    plot_policy_heatmap(pi, save_path)

if __name__ == "__main__":
    main()
