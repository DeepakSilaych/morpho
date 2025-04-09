"""Utilities for visualizing BPE merge operations."""

from typing import List, Dict, Tuple, Optional
import os
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
    import networkx as nx
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False


def check_visualization_dependencies():
    """Check if visualization dependencies are installed."""
    if not VISUALIZATION_AVAILABLE:
        print("Visualization dependencies are not installed.")
        print("Install them with: pip install matplotlib networkx")
        return False
    return True


def create_merge_graph(merges: Dict[Tuple[str, str], str], 
                      tokens: Optional[List[str]] = None, 
                      max_nodes: int = 50) -> nx.DiGraph:
    """Create a directed graph of BPE merge operations.
    
    Args:
        merges: Dictionary mapping token pairs to merged tokens
        tokens: Optional list of tokens to include in the graph
        max_nodes: Maximum number of nodes to include in the graph
        
    Returns:
        NetworkX DiGraph object
    """
    if not check_visualization_dependencies():
        return None
    
    G = nx.DiGraph()
    
    # Add all nodes and edges from merges
    added_nodes = set()
    
    # Filter by specific tokens if provided
    if tokens:
        filtered_merges = {}
        tokens_set = set(tokens)
        for (token1, token2), merged in merges.items():
            if (token1 in tokens_set or token2 in tokens_set or merged in tokens_set):
                filtered_merges[(token1, token2)] = merged
        merges_to_use = filtered_merges
    else:
        merges_to_use = merges
    
    # Limit to max_nodes if needed
    if max_nodes and len(merges_to_use) > max_nodes:
        merges_to_use = dict(list(merges_to_use.items())[:max_nodes])
    
    # Add edges for each merge
    for (token1, token2), merged in merges_to_use.items():
        if token1 not in added_nodes:
            G.add_node(token1, size=len(token1))
            added_nodes.add(token1)
            
        if token2 not in added_nodes:
            G.add_node(token2, size=len(token2))
            added_nodes.add(token2)
            
        if merged not in added_nodes:
            G.add_node(merged, size=len(merged))
            added_nodes.add(merged)
            
        G.add_edge(token1, merged)
        G.add_edge(token2, merged)
    
    return G


def plot_merge_graph(merges: Dict[Tuple[str, str], str], 
                    tokens: Optional[List[str]] = None, 
                    max_nodes: int = 50,
                    figsize: Tuple[int, int] = (12, 10),
                    title: str = "BPE Merge Operations",
                    output_path: Optional[str] = None):
    """Plot a graph of BPE merge operations.
    
    Args:
        merges: Dictionary mapping token pairs to merged tokens
        tokens: Optional list of tokens to include in the graph
        max_nodes: Maximum number of nodes to include in the graph
        figsize: Figure size
        title: Plot title
        output_path: Optional path to save the plot
    """
    if not check_visualization_dependencies():
        return
    
    G = create_merge_graph(merges, tokens, max_nodes)
    if not G:
        return
    
    plt.figure(figsize=figsize)
    
    # Define node sizes based on token length
    node_sizes = [G.nodes[node]['size'] * 200 for node in G.nodes]
    
    # Use different colors based on node type (source, merged)
    node_colors = []
    for node in G.nodes:
        if G.in_degree(node) == 0:  # Source token
            node_colors.append('lightblue')
        else:  # Merged token
            node_colors.append('lightgreen')
    
    # Draw the graph
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_size=node_sizes, 
            node_color=node_colors, font_size=10, 
            font_family='sans-serif')
    
    plt.title(title)
    
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path)
    
    plt.show()


def visualize_token_segmentation(text: str, 
                                tokens: List[str], 
                                figsize: Tuple[int, int] = (12, 3),
                                colors: Optional[List[str]] = None,
                                title: str = "Token Segmentation",
                                output_path: Optional[str] = None):
    """Visualize how a text is segmented into tokens.
    
    Args:
        text: Original text
        tokens: List of tokens from the tokenizer
        figsize: Figure size
        colors: Optional list of colors for tokens
        title: Plot title
        output_path: Optional path to save the plot
    """
    if not check_visualization_dependencies():
        return
    
    # Determine token positions in the original text
    positions = []
    current_pos = 0
    reconstructed = ""
    
    for token in tokens:
        # Special handling for WordPiece-style tokens
        if token.startswith('##'):
            clean_token = token[2:]
            offset = 0  # No space before continuation
        # Special handling for SentencePiece-style tokens
        elif token.startswith('▁'):
            clean_token = token[1:]
            offset = 1  # Space before new word
        # Handle special tokens
        elif token in ['<unk>', '<pad>', '<s>', '</s>', '[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']:
            clean_token = token
            offset = 0
        else:
            clean_token = token
            offset = 0
            # Add space if this is not the first token and the previous doesn't end with space
            if reconstructed and not reconstructed.endswith(' '):
                offset = 1
        
        # Skip special tokens in visualization
        if clean_token in ['<unk>', '<pad>', '<s>', '</s>', '[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']:
            continue
        
        # Find position in text
        search_pos = current_pos
        while True:
            if offset > 0 and search_pos < len(text) and text[search_pos] == ' ':
                search_pos += offset
                offset = 0
            
            if search_pos + len(clean_token) <= len(text) and text[search_pos:search_pos + len(clean_token)] == clean_token:
                positions.append((search_pos, search_pos + len(clean_token), clean_token))
                current_pos = search_pos + len(clean_token)
                reconstructed += clean_token
                break
            
            search_pos += 1
            if search_pos >= len(text):
                # Token not found in text (could be due to normalization)
                break
    
    # Plot the segmentation
    plt.figure(figsize=figsize)
    
    # Default colors
    if not colors:
        colors = ['#f72585', '#7209b7', '#3a0ca3', '#4361ee', '#4cc9f0']
    
    y_pos = 0
    for i, (start, end, token) in enumerate(positions):
        color = colors[i % len(colors)]
        plt.axvspan(start, end, alpha=0.3, color=color)
        plt.text((start + end) / 2, y_pos + 0.5, token, 
                ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Plot the original text
    for i, char in enumerate(text):
        plt.text(i + 0.5, y_pos + 1.5, char, ha='center', va='center', fontsize=12)
    
    plt.xlim(0, len(text))
    plt.ylim(0, 2)
    plt.title(title)
    plt.axis('off')
    
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path)
    
    plt.show() 