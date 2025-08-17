import sys
import os
from typing import List, Tuple, Dict, Optional
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from jtnn_utils.mol_tree import MolTree, MolTreeNode


def dfs_serialize_tree(
        tree: MolTree,
) -> Tuple[List[MolTreeNode], List[int], List[int], List[int]]:
    """
    Serializes a MolTree using a pre-order DFS traversal and computes structural features.

    This implementation uses a more efficient, single-stack iterative approach. It
    tracks visited children for each node to correctly manage the traversal and
    the calculation of backtrack relations.

    Args:
        tree (MolTree): The molecule tree to serialize.

    Returns:
        A tuple containing:
        - nodes_in_order (List[MolTreeNode]): List of nodes in pre-order.
        - relations (List[int]): List of backtrack relations `R`. `R[i]` is the
          backtrack count from node `i` to find the parent of node `i+1`.
        - layer_numbers (List[int]): The depth of each node in the tree.
        - parent_positions (List[int]): The index of each node's parent in the
          `nodes_in_order` list. The root's parent position is 0.
    """
    if not tree.nodes:
        return [], [], [], []

    root = tree.nodes[0]

    # --- Pre-computation ---
    # Create adjacency list and parent map for efficient lookup
    adj: Dict[MolTreeNode, List[MolTreeNode]] = {node: [] for node in tree.nodes}
    parent_map: Dict[MolTreeNode, MolTreeNode] = {}
    q = [root]
    visited_for_adj = {root}
    head = 0
    while head < len(q):
        u = q[head]
        head += 1
        for v in u.neighbors:
            if v not in visited_for_adj:
                visited_for_adj.add(v)
                adj[u].append(v)
                adj[v].append(u) 
                parent_map[v] = u
                q.append(v)

    # Ensure adjacency list only contains children for traversal
    for node in tree.nodes:
        adj[node] = [child for child in adj[node] if parent_map.get(child) == node]

    # --- Outputs ---
    nodes_in_order: List[MolTreeNode] = []
    relations: List[int] = []
    layer_numbers: List[int] = []
    parent_positions: List[int] = []

    # --- DFS State ---
    # stack item: (node, depth)
    stack: List[Tuple[MolTreeNode, int]] = [(root, 0)]
    # Tracks the current path from the root for relation calculation
    path_stack: List[MolTreeNode] = []
    node_to_idx: Dict[MolTreeNode, int] = {}

    while stack:
        current_node, depth = stack.pop()
        # Calculate relation `r` based on the transition from the previous node
        if nodes_in_order:
            parent_of_current = parent_map.get(current_node)
            backtrack_count = 0
            # Pop from the path until the parent is at the top
            while path_stack and path_stack[-1] != parent_of_current:
                path_stack.pop()
                backtrack_count += 1
            relations.append(backtrack_count)

        # Record features for the current node
        nodes_in_order.append(current_node)
        node_to_idx[current_node] = len(nodes_in_order) - 1
        layer_numbers.append(depth)

        parent = parent_map.get(current_node)
        if parent:
            parent_positions.append(node_to_idx[parent])
        else:
            # Root's parent can be considered itself at position 0
            parent_positions.append(0)

        path_stack.append(current_node)

        # Add children to the main stack in reverse order for correct pre-order traversal
        children = adj.get(current_node, [])
        for child in reversed(children):
            stack.append((child, depth + 1))
    relations.insert(0, -1)
    return nodes_in_order, relations, layer_numbers, parent_positions
