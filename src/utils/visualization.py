import matplotlib.pyplot as plt
import numpy as np
from pygsp import graphs


def visualize_coarsening_results(G_original: graphs.Graph, G_coarsened: graphs.Graph, method_name: str):
    """
    Visualize the original and coarsened graphs
    """
    # Ensure both graphs have coordinates
    if not hasattr(G_original, 'coords') or G_original.coords is None:
        print("Setting coordinates for original graph...")
        try:
            G_original.set_coordinates('spring', dim=2)
        except:
            G_original.set_coordinates('random2D')
    
    if not hasattr(G_coarsened, 'coords') or G_coarsened.coords is None:
        print("Setting coordinates for coarsened graph...")
        try:
            G_coarsened.set_coordinates('spring', dim=2)
        except:
            G_coarsened.set_coordinates('random2D')
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    
    axes[0].set_title(f'Original Cora Graph\n{G_original.N} nodes, {G_original.Ne} edges')
    G_original.plot(ax=axes[0], vertex_size=15)
    
    # Coarsened graph  
    axes[1].set_title(f'Coarsened Graph ({method_name})\n{G_coarsened.N} nodes, {G_coarsened.Ne} edges')
    G_coarsened.plot(ax=axes[1], vertex_size=25)
    
    # Reduction comparison (bar chart)
    axes[2].bar(['Original', 'Coarsened'], [G_original.N, G_coarsened.N], 
               color=['lightblue', 'lightcoral'], alpha=0.7)
    axes[2].set_ylabel('Number of Nodes')
    axes[2].set_title(f'Size Reduction\nRatio: {G_original.N / G_coarsened.N:.2f}x')
    axes[2].grid(True, alpha=0.3)
    
    # Add text annotations
    for i, v in enumerate([G_original.N, G_coarsened.N]):
        axes[2].text(i, v + 20, str(v), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def visualize_graph_statistics(G_original, G_coarsened, method_name):
    """
    Statistics on graphs
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Size comparison
    sizes = [G_original.N, G_coarsened.N]
    labels = ['Original', 'Coarsened']
    colors = ['lightblue', 'lightcoral']
    
    axes[0,0].bar(labels, sizes, color=colors, alpha=0.7)
    axes[0,0].set_ylabel('Number of Nodes')
    axes[0,0].set_title(f'Graph Size Comparison ({method_name})')
    axes[0,0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(sizes):
        axes[0,0].text(i, v + max(sizes)*0.01, str(v), ha='center', va='bottom', fontweight='bold')
    
    # 2. Edge density comparison
    edge_densities = [2*G_original.Ne/(G_original.N*(G_original.N-1)), 
                     2*G_coarsened.Ne/(G_coarsened.N*(G_coarsened.N-1))]
    
    axes[0,1].bar(labels, edge_densities, color=colors, alpha=0.7)
    axes[0,1].set_ylabel('Edge Density')
    axes[0,1].set_title('Edge Density Comparison')
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Degree distribution comparison
    degrees_orig = np.array([G_original.W[i].sum() for i in range(G_original.N)])
    degrees_coars = np.array([G_coarsened.W[i].sum() for i in range(G_coarsened.N)])
    
    axes[1,0].hist(degrees_orig, bins=30, alpha=0.7, label='Original', density=True, color='lightblue')
    axes[1,0].hist(degrees_coars, bins=20, alpha=0.7, label='Coarsened', density=True, color='lightcoral')
    axes[1,0].set_xlabel('Degree')
    axes[1,0].set_ylabel('Density')
    axes[1,0].set_title('Degree Distribution')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Summary statistics
    stats_text = f"""
    Reduction Statistics ({method_name}):
    
    Original Graph:
    • Nodes: {G_original.N:,}
    • Edges: {G_original.Ne:,}
    • Avg Degree: {degrees_orig.mean():.2f}
    • Max Degree: {degrees_orig.max()}
    
    Coarsened Graph:
    • Nodes: {G_coarsened.N:,}
    • Edges: {G_coarsened.Ne:,}
    • Avg Degree: {degrees_coars.mean():.2f}
    • Max Degree: {degrees_coars.max()}
    
    Reduction Ratios:
    • Node Reduction: {G_original.N/G_coarsened.N:.2f}x
    • Edge Reduction: {G_original.Ne/G_coarsened.Ne:.2f}x
    """
    
    axes[1,1].text(0.05, 0.95, stats_text, transform=axes[1,1].transAxes, 
                   verticalalignment='top', fontfamily='monospace', fontsize=10)
    axes[1,1].set_xlim(0, 1)
    axes[1,1].set_ylim(0, 1)
    axes[1,1].axis('off')
    
    plt.tight_layout()
    plt.show()
    """
    Compare spectral properties of original and coarsened graphs
    """
    print("Spectral Analysis:")
    
    # Compute eigenvalues
    G_original.compute_fourier_basis()
    G_coarsened.compute_fourier_basis()
    
    # Plot eigenvalue comparison
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(G_original.e[:50], 'b-', label='Original', alpha=0.7)
    plt.plot(G_coarsened.e[:min(50, len(G_coarsened.e))], 'r-', label='Coarsened', alpha=0.7)
    plt.xlabel('Index')
    plt.ylabel('Eigenvalue')
    plt.title('Eigenvalue Comparison (first 50)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(G_original.e, bins=50, alpha=0.7, label='Original', density=True)
    plt.hist(G_coarsened.e, bins=30, alpha=0.7, label='Coarsened', density=True)
    plt.xlabel('Eigenvalue')
    plt.ylabel('Density')
    plt.title('Eigenvalue Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
def plot_training_curves(train_losses, val_losses, val_accuracies):
    """Plot training curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curves
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy curve
    ax2.plot(val_accuracies, label='Val Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
