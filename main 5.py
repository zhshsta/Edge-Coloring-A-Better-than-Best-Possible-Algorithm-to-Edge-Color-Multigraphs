#If the graph is too big you will have to change the line q = next(iter(set(range(1, 41)).difference(colors_in_use)))  # I use max 40 colors
# pip install networkx matplotlib
# import collections
# import itertools
# import numpy as np
import webcolors
import networkx as nx
import matplotlib.pyplot as plt
import random
from matplotlib.patches import FancyArrowPatch
""" Read Input File"""


def read_edges_from_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Extract edges from lines
    edges = [tuple(map(int, line.split(':')[0].split())) for line in lines]
    return edges


def create_adjacency_matrix(edges):
    # Find the number of nodes by locating the maximum labeled node
    n = max(max(edge) for edge in edges)

    # Initialize the matrix with zeros
    matrix = [[0 for _ in range(n)] for _ in range(n)]

    # Populate the matrix
    for (i, j) in edges:
        matrix[i - 1][j - 1] += 1
        if i != j:  # Avoid double counting for self-loops
            matrix[j - 1][i - 1] += 1

    return matrix


def print_matrix(matrix):
    for row in matrix:
        print(" ".join(map(str, row)))


def create_edge_dict_from_matrix(matrix):
    edge_dict = {}
    for i in range(len(matrix)):
        for j in range(i, len(matrix)):
            for count in range(matrix[i][j]):
                edge_dict[(i, j, count)] = None  # Initializing with None (no color assigned yet)
    return edge_dict


def color_to_edges_dict(edge_dict):
    color_edge_dict = {}
    for (u, v, edge_count), color in edge_dict.items():
        if color not in color_edge_dict:
            color_edge_dict[color] = []
        color_edge_dict[color].append((u, v, edge_count))
    return color_edge_dict


def get_colored_edges(edge_colors):
    """
    Returns a list of tuples representing each edge and its color.
    [(u, v, edge_count, color), ...]
    """
    return [(u, v, edge_count, color) for (u, v, edge_count), color in edge_colors.items()]


""" Coloring using dfs """


def is_valid_edge_color(u, v, count, color, edge_colors, matrix):
    def get_incident_edges(vertex):
        edges = []
        for i in range(len(matrix)):
            if matrix[vertex][i] > 0:
                for edge_count in range(matrix[vertex][i]):
                    edges.append((vertex, i, edge_count))
                    if vertex != i:  # Avoid double counting for self-loops
                        edges.append((i, vertex, edge_count))
        return edges

    # Get edges incident on u and v
    incident_edges_u = get_incident_edges(u)
    incident_edges_v = get_incident_edges(v)

    # Check if the color already exists on incident edges of u
    for edge in incident_edges_u:
        if edge != (u, v, count) and edge_colors.get(edge) == color:
            return False

    # Check if the color already exists on incident edges of v
    for edge in incident_edges_v:
        if edge != (v, u, count) and edge_colors.get(edge) == color:
            return False

    return True


def color_edges_dfs(current_edge, matrix, edge_dict, num_colors, edge_manager):
    print("Entering color_edges_dfs")

    if current_edge == len(edge_dict):
        return True

    u, v, edge_count = list(edge_dict.keys())[current_edge]
    print(f"Trying to color edge {u}-{v} (count: {edge_count})")

    for color in range(1, num_colors + 1):
        #print(f"  Testing color {color} for edge {u}-{v} (count: {edge_count})")

        if is_valid_edge_color(u, v, edge_count, color, edge_dict, matrix):
            #print(f"  Assigning color {color} to edge {u}-{v} (count: {edge_count})")
            edge_dict[(u, v, edge_count)] = color


            # Update the EdgeManager instance with the newly determined color
            edge_manager.color_edge(u, v, color, edge_count)

            if color_edges_dfs(current_edge + 1, matrix, edge_dict, num_colors, edge_manager):
                return True
            edge_dict[(u, v, edge_count)] = None
            #print(f"  Removing color from edge {u}-{v} (count: {edge_count})")

            edge_manager.color_edge(u, v, None, edge_count)  # Reset the edge color in EdgeManager to None as well
    print("exiting color_edges_dfs")

    return False


def dfs_edge_coloring(matrix, num_colors, edge_dict, edge_manager):
    print("Entering dfs_edge_coloring")
    success = color_edges_dfs(0, matrix, edge_dict, num_colors, edge_manager)
    if success:
        for (u, v, count), color in edge_dict.items():
            edge_manager.color_edge(u, v, color, count)
    else:
        print("Failed in dfs_edge_coloring")

    return success


# def delete_edge_from_dict(u, v, count, edge_dict):
#     # Adjust the count of subsequent edges and shift their colors down
#     color_to_keep = None
#     if (u, v, count) in edge_dict:
#         color_to_keep = edge_dict[(u, v, count)]
#         del edge_dict[(u, v, count)]
#
#         next_count = count
#         while (u, v, next_count + 1) in edge_dict:
#             edge_dict[(u, v, next_count)] = edge_dict[(u, v, next_count + 1)]
#             next_count += 1
#         if (u, v, next_count) in edge_dict:
#             del edge_dict[(u, v, next_count)]
#     elif (v, u, count) in edge_dict:
#         color_to_keep = edge_dict[(v, u, count)]
#         del edge_dict[(v, u, count)]
#
#         next_count = count
#         while (v, u, next_count + 1) in edge_dict:
#             edge_dict[(v, u, next_count)] = edge_dict[(v, u, next_count + 1)]
#             next_count += 1
#         if (v, u, next_count) in edge_dict:
#             del edge_dict[(v, u, next_count)]
#     return color_to_keep


"""bi-directional mapping"""


class EdgeManager:
    def __init__(self, matrix):
        self.matrix = matrix
        self.edge_to_color_dict = self._initialize_edge_dict()
        self.color_to_edge_dict = {}  # This will be a bi-mapping with edge_to_color_dict
        self.uncolored_edges = list(self.edge_to_color_dict.keys())

    def _initialize_edge_dict(self):
        edge_dict = {}
        for i in range(len(self.matrix)):
            for j in range(i, len(self.matrix)):
                for count in range(self.matrix[i][j]):
                    edge_key = (i, j, count)
                    edge_dict[edge_key] = None
        #print("Initial edge_to_color_dict:", edge_dict)
        return edge_dict

    def color_edge(self, u, v, color, count=0):
        edge_key_1 = (u, v, count)
        edge_key_2 = (v, u, count)

        # Determine the actual edge key present in the dict
        edge_key = edge_key_1 if edge_key_1 in self.edge_to_color_dict else edge_key_2

        # Remove the old color assignment from color_to_edge_dict
        old_color = self.edge_to_color_dict.get(edge_key)
        if old_color and old_color in self.color_to_edge_dict:
            self.color_to_edge_dict[old_color].remove(edge_key)
            if not self.color_to_edge_dict[old_color]:
                del self.color_to_edge_dict[old_color]

        # Update edge_to_color_dict
        if color:
            was_uncolored = self.edge_to_color_dict[edge_key] is None  # Check if it was previously uncolored
            self.edge_to_color_dict[edge_key] = color
            self.color_to_edge_dict.setdefault(color, []).append(edge_key)
            if was_uncolored:
                try:
                    self.uncolored_edges.remove(edge_key)
                except ValueError:
                    print(f"Attempted to remove {edge_key} which is not in the list.")
        else:
            self.edge_to_color_dict[edge_key] = None
            if edge_key not in self.uncolored_edges:
                self.uncolored_edges.append(edge_key)

    def delete_edge(self, u, v, count=0):
        edge_key_1 = (u, v, count)
        edge_key_2 = (v, u, count)

        # Print the current status of dictionaries for debugging
        #print("Before removal:")
        #print("edge_to_color_dict:", self.edge_to_color_dict)
        #print("color_to_edge_dict:", self.color_to_edge_dict)

        # Explicitly print the edge we're trying to remove for debugging
        #print(f"Attempting to remove edge: {edge_key_1}")

        # Remove edge from matrix
        self.matrix[u][v] -= 1
        self.matrix[v][u] -= 1

        # Remove edge from edge_to_color_dict
        if edge_key_1 in self.edge_to_color_dict:
            color = self.edge_to_color_dict[edge_key_1]
            del self.edge_to_color_dict[edge_key_1]
        elif edge_key_2 in self.edge_to_color_dict:
            color = self.edge_to_color_dict[edge_key_2]
            del self.edge_to_color_dict[edge_key_2]
        else:
            print(f"Edge {edge_key_1} not found in edge_to_color_dict!")
            return

        # Remove edge from color_to_edge_dict
        if color in self.color_to_edge_dict:
            if edge_key_1 in self.color_to_edge_dict[color]:
                self.color_to_edge_dict[color].remove(edge_key_1)
            elif edge_key_2 in self.color_to_edge_dict[color]:
                self.color_to_edge_dict[color].remove(edge_key_2)
            if not self.color_to_edge_dict[color]:  # if no edge remains for a color, remove the color entry
                del self.color_to_edge_dict[color]

        # Print the updated status of dictionaries for debugging
        #print("After removal:")
        #print("edge_to_color_dict:", self.edge_to_color_dict)
        #print("color_to_edge_dict:", self.color_to_edge_dict)

    def find_edge_count(self, u, v, color):
        """
        Finds the count of an edge between two vertices with a specific color.
        Returns the count if found, otherwise returns None.
        """
        # Check if the color exists in the color_to_edge_dict
        if color in self.color_to_edge_dict:
            for edge in self.color_to_edge_dict[color]:
                edge_u, edge_v, edge_count = edge
                # Check if the edge matches the input vertices
                if (edge_u == u and edge_v == v) or (edge_u == v and edge_v == u):
                    return edge_count
        return None

    def uncolor_edge(self, u, v, count=0):
        """
        Removes the color of an edge, effectively setting it to None.
        """
        edge_key_1 = (u, v, count)
        edge_key_2 = (v, u, count)

        # Determine the actual edge key present in the dict, possible no need for check
        edge_key = edge_key_1 if edge_key_1 in self.edge_to_color_dict else edge_key_2

        # Remove the color assignment from color_to_edge_dict
        color = self.edge_to_color_dict.get(edge_key)
        if color is not None:
            self.color_to_edge_dict[color].remove(edge_key)
            if not self.color_to_edge_dict[color]:
                del self.color_to_edge_dict[color]

        # Update edge_to_color_dict to None
        self.edge_to_color_dict[edge_key] = None
        if edge_key not in self.uncolored_edges:
            self.uncolored_edges.append(edge_key)

        print(f"Edge {edge_key} is now uncolored.")

    # def delete_edge(self, u, v, count):
    #     # Decrement the matrix count
    #     self.matrix[u][v] -= 1
    #     self.matrix[v][u] -= 1
    #
    #     # Shift the edges down
    #     current_count = count
    #     while (u, v, current_count + 1) in self.edge_dict:
    #         # Shifting the edges down
    #         self.edge_dict[(u, v, current_count)] = self.edge_dict[(u, v, current_count + 1)]
    #         self.edge_dict[(v, u, current_count)] = self.edge_dict[(v, u, current_count + 1)]
    #
    #         # Increase the current_count
    #         current_count += 1
    #
    #     # Now, delete the highest count edge for that pair of nodes
    #     if (u, v, current_count) in self.edge_dict:
    #         del self.edge_dict[(u, v, current_count)]
    #     if (v, u, current_count) in self.edge_dict:
    #         del self.edge_dict[(v, u, current_count)]
    #
    #     # Recolor the edges after deletion
    #     success = dfs_edge_coloring(self.matrix, 10, self.edge_dict)
    #     if not success:
    #         raise ValueError("Failed to recolor the edges after deletion.")

    # def delete_single_edge(self, u, v, count):
    #     # Check if the edge exists in the dictionary
    #     if (u, v, count) in self.edge_dict:
    #         del self.edge_dict[(u, v, count)]
    def get_colors(self, vertex):
        colors = set()
        for edge, color in self.edge_to_color_dict.items():
            u, v, _ = edge
            if u == vertex or v == vertex:
                if color:  # ensuring that the edge has been colored
                    colors.add(color)
        return colors

    def get_matrix(self):
        return self.matrix

    def get_edge_color_dict(self):
        return self.edge_to_color_dict

    def get_color_edge_dict(self):
        return self.color_to_edge_dict

    def get_uncolored_edges(self):
        return self.uncolored_edges


"""further functions"""
# def ab_subgraph(edge_manager, a, b):
#     """
#     Extracts the subgraph consisting of edges with colors a and b from the edge_manager.
#     Returns the subgraph's connected components and identifies them as paths or cycles.
#     """
#     print(f"Extracting subgraph with colors: {a} and {b}")
#     color_to_edge_dict = edge_manager.get_color_edge_dict()
#     subG = nx.MultiGraph()
#     print("Joined ab_subgraph")
#
#     # Add uncolored edges to subG
#     uncolored_edges = [edge for edge, color in color_to_edge_dict.items() if color is None]
#     start_nodes = set()
#     for u, v, count in uncolored_edges:
#         start_nodes.add(u)
#         start_nodes.add(v)
#         subG.add_edge(u, v, color=None)  # Add uncolored edges to subG
#     print(f"Added {len(uncolored_edges)} uncolored edges to the subgraph.")
#
#     # Extract the subgraph that only contains edges of color 'a' and 'b'
#     for color in [a, b]:
#         if color in color_to_edge_dict:
#             edges = color_to_edge_dict[color]
#             for u, v, count in edges:
#                 subG.add_edge(u, v, color=color)
#             print(f"Added {len(edges)} edges of color {color} to the subgraph.")
#
#     print(f"Subgraph nodes: {list(subG.nodes())}")
#     print(f"Subgraph edges: {list(subG.edges(data=True))}")
#
#     # Create an empty adjacency matrix for the subgraph
#     n = max(subG.nodes()) + 1  # Ensure the matrix is large enough for the highest node number
#     adj_matrix = [[0 for _ in range(n)] for _ in range(n)]
#
#     # Populate the adjacency matrix for the subgraph
#     for u, v in subG.edges():
#         adj_matrix[u][v] += 1
#         if u != v:  # Avoid double-counting if it's a self-loop
#             adj_matrix[v][u] += 1
#
#     print("Adjacency matrix populated:")
#     print_matrix(adj_matrix)
#
#     components = []
#     seen_nodes = set()
#
#     # Attempt to identify paths and cycles within the subgraph
#     print(f"Attempting to identify paths and cycles in the subgraph...")
#     for start_node in start_nodes:
#         if start_node in seen_nodes:
#             continue
#
#         # Use a stack for DFS and a dictionary to remember paths to each node
#         stack = [(start_node, None)]  # Each entry is a tuple (node, parent)
#         path_to_node = {start_node: [start_node]}
#
#         while stack:
#             current_node, parent = stack.pop()
#
#             if current_node in seen_nodes:
#                 # We have returned to a node already seen; a cycle has been detected
#                 cycle_path = path_to_node[current_node] + [current_node]
#                 components.append(('cycle', cycle_path))
#                 print(f"Cycle found: {cycle_path}")
#                 continue
#
#             seen_nodes.add(current_node)
#             for neighbor in subG.neighbors(current_node):
#                 if neighbor == parent:
#                     # Don't go back to the parent node
#                     continue
#                 if neighbor in path_to_node:
#                     # This neighbor has been visited via a different path; a cycle has been detected
#                     cycle_path = path_to_node[neighbor] + [current_node]
#                     components.append(('cycle', cycle_path))
#                     print(f"Cycle found: {cycle_path}")
#                     continue
#                 # Update the path to the neighbor
#                 path_to_node[neighbor] = path_to_node[current_node] + [neighbor]
#                 stack.append((neighbor, current_node))
#
#             if len(subG[current_node]) == (1 if parent is None else 2):
#                 # A leaf node in the subgraph is found, signifying the end of a path
#                 components.append(('path', path_to_node[current_node]))
#                 print(f"Path found: {path_to_node[current_node]}")
#
#     print("Finished processing subgraph.")
#     return components, adj_matrix


def apath_for_component(edge_manager, start_vertex, color_a, color_b):
    """
    the difference here is that we don't want the path to have len > 2
    """
    print("Joined apath")
    subG = nx.MultiGraph()
    color_to_edge_dict = edge_manager.get_color_edge_dict()
    #print(color_to_edge_dict)
    print("starting vertex :color_a color_b ", start_vertex, color_a, color_b)

    # Extract edges with color_a and color_b into the subgraph
    for color in [color_a, color_b]:
        if color in color_to_edge_dict:
            for u, v, count in color_to_edge_dict[color]:
                subG.add_edge(u, v, color=color)

    seen_nodes = set()
    stack = [(start_vertex, [start_vertex], None)]  # Start traversal from 'start_vertex'

    while stack:
        current_node, current_path, last_color = stack.pop()

        if current_node in seen_nodes:
            continue  # Skip already seen nodes

        seen_nodes.add(current_node)

        path_continued = False
        for neighbor in subG[current_node]:
            edge_color = subG[current_node][neighbor][0]['color']

            # Check if the neighbor is the starting vertex and is not the next immediate neighbor
            if neighbor == start_vertex and len(current_path) > 2:
                continue  # This would form a loop

            # Check for color alternation
            if edge_color in [color_a, color_b] and edge_color != last_color:
                new_path = current_path + [neighbor]
                stack.append((neighbor, new_path, edge_color))
                #print(f"Extended path to {new_path}")
                path_continued = True  # Mark that the path has been extended

        # If no valid extension is found, check if current_path is a valid path
        if not path_continued:  # Length check ensures at least one edge in the path
            print(f"Valid path found: {current_path}")
            return current_path

    print("No path found after processing all possibilities.")
    return []  # Return an empty list if no path is found


def apath_or_cycles(edge_manager, start_vertex, color_a, color_b):
    """
    Finds a path or a cycle starting from 'start_vertex' where the edges are alternately colored with 'color_a' and 'color_b'.
    :return: The path or cycle found as a list of vertices, or an empty list if no path or cycle is found.
    """
    print("Joined apath_with_cycles")
    subG = nx.MultiGraph()
    color_to_edge_dict = edge_manager.get_color_edge_dict()
    #print(color_to_edge_dict)
    #print("starting vertex :color_a color_b ", start_vertex, color_a, color_b)

    # Extract edges with color_a and color_b into the subgraph
    for color in [color_a, color_b]:
        if color in color_to_edge_dict:
            for u, v, count in color_to_edge_dict[color]:
                subG.add_edge(u, v, color=color)

    seen_nodes = set()
    stack = [(start_vertex, [start_vertex], None)]  # Start traversal from 'start_vertex'

    while stack:
        current_node, current_path, last_color = stack.pop()

        # Check if the current node has been visited before, indicating a cycle
        if current_node in current_path[:-1]:
            cycle_start_index = current_path.index(current_node)
            cycle = current_path[cycle_start_index:]
            print(f"Cycle found: {cycle}")
            return cycle

        seen_nodes.add(current_node)

        for neighbor in subG[current_node]:
            edge_color = subG[current_node][neighbor][0]['color']

            # Check for color alternation
            if edge_color in [color_a, color_b] and edge_color != last_color:
                new_path = current_path + [neighbor]
                stack.append((neighbor, new_path, edge_color))
                #print(f"Extended path to {new_path}")

    print("No path or cycle found after processing all possibilities.")
    return []  # Return an empty list if no path or cycle is found


def apath(edge_manager, start_vertex, color_a, color_b):
    """
    Finds a path starting from 'start_vertex' where the edges are alternately colored with 'color_a' and 'color_b'.
    :return: The path found as a list of vertices, or an empty list if no path is found.
    """
    print("Joined apath")
    subG = nx.MultiGraph()
    color_to_edge_dict = edge_manager.get_color_edge_dict()
    #print(color_to_edge_dict)
    print("starting vertex :color_a color_b ", start_vertex, color_a, color_b)

    # Extract edges with color_a and color_b into the subgraph
    for color in [color_a, color_b]:
        if color in color_to_edge_dict:
            for u, v, count in color_to_edge_dict[color]:
                subG.add_edge(u, v, color=color)

    seen_nodes = set()
    stack = [(start_vertex, [start_vertex], None)]  # Start traversal from 'start_vertex'

    # Check if there are edges with color_a and color_b starting from the start vertex
    if color_a in color_to_edge_dict and color_b in color_to_edge_dict:
        edges_with_color_a = [(u, v, count) for u, v, count in color_to_edge_dict[color_a] if
                              u == start_vertex or v == start_vertex]
        edges_with_color_b = [(u, v, count) for u, v, count in color_to_edge_dict[color_b] if
                              u == start_vertex or v == start_vertex]
        if edges_with_color_b or edges_with_color_a is None:
            return

    while stack:
        current_node, current_path, last_color = stack.pop()

        if current_node in seen_nodes:
            continue  # Skip already seen nodes

        seen_nodes.add(current_node)

        path_continued = False

        edges_with_color_a = [(u, v, count) for u, v, count in color_to_edge_dict[color_a] if
                              u == start_vertex or v == start_vertex]
        edges_with_color_b = [(u, v, count) for u, v, count in color_to_edge_dict[color_b] if
                              u == start_vertex or v == start_vertex]
        #print(edges_with_color_b, edges_with_color_a)
        if not edges_with_color_b:
            continue
        if not edges_with_color_a :
            continue

        for neighbor in subG[current_node]:
            edge_color = subG[current_node][neighbor][0]['color']

            # Check if the neighbor is the starting vertex and is not the next immediate neighbor
            if neighbor == start_vertex and len(current_path) > 2:
                continue  # This would form a loop

            # Check for color alternation
            if edge_color in [color_a, color_b] and edge_color != last_color:
                new_path = current_path + [neighbor]
                stack.append((neighbor, new_path, edge_color))
                #print(f"Extended path to {new_path}")
                path_continued = True  # Mark that the path has been extended

        # If no valid extension is found, check if current_path is a valid path
        if not path_continued and len(current_path) > 2:  # Length check ensures at least one edge in the path
            print(f"Valid path found: {current_path}")
            return current_path

    print("No path found after processing all possibilities.")
    return []  # Return an empty list if no path is found


def ab_subgraph(edge_manager, a, b, critical_edge_key):
    """
    Extracts the subgraph consisting of edges with colors a and b from the edge_manager.
    Returns the subgraph's connected components and identifies them as paths or cycles.
    """
    color_to_edge_dict = edge_manager.get_color_edge_dict()
    subG = nx.MultiGraph()
    print("Joined ab_subgraph")
    #print("critical edge", critical_edge_key)

    # # The idea for having a starting vertex for the paths. The edge we add is an uncolored one.
    # uncolored_edges = [edge for edge, color in color_to_edge_dict.items() if color is None]
    # start_nodes = set()
    # for u, v, count in uncolored_edges:
    #     start_nodes.add(u)
    #     start_nodes.add(v)
    #     subG.add_edge(u, v, color=None)  # Add uncolored edges to subG
    #print(color_to_edge_dict)

    # Extract the subgraph that only contains edges of color 'a' and 'b'
    #print("Adding edges to subG:")
    for color in [a, b]:
        if color in color_to_edge_dict:
            for u, v, count in color_to_edge_dict[color]:
                subG.add_edge(u, v, color=color)
                #print(f"Added edge ({u}, {v}) with color {color}")

    # Display the nodes in subG
    #print("Nodes in subG:", subG.nodes())

    # # Create an empty adjacency matrix for the subgraph
    # n = len(subG.nodes())
    # adj_matrix = [[0 for _ in range(n)] for _ in range(n)]
    #
    # for edge in subG.edges():
    #     u, v = edge[:2]  # Extract nodes from edge tuple
    #     print(f"u={u}, v={v}, adj_matrix dimensions={len(adj_matrix)}x{len(adj_matrix[0])}")
    #     if 0 <= u < len(adj_matrix) and 0 <= v < len(adj_matrix[0]):
    #         adj_matrix[u][v] += 1
    #         if u != v:  # Avoid double-counting if it's a self-loop
    #             adj_matrix[v][u] += 1
    #     else:
    #         print(f"Error: Invalid indices u={u} and v={v} for adj_matrix dimensions={len(adj_matrix)}x{len(adj_matrix[0])}")
    # Determine the size of the adjacency matrix
    max_vertex = max([max(edge[:2]) for edge in subG.edges()])
    n = max_vertex + 1  # Adjusting for zero-based indexing

    # Initialize an empty adjacency matrix
    adj_matrix = [[0 for _ in range(n)] for _ in range(n)]

    # Fill the adjacency matrix
    for edge in subG.edges():
        u, v = edge[:2]
        if 0 <= u < n and 0 <= v < n:
            adj_matrix[u][v] += 1
            if u != v:
                adj_matrix[v][u] += 1
        else:
            print(f"Error: Invalid indices u={u} and v={v} for adj_matrix dimensions={n}x{n}")

    #print_matrix(adj_matrix)
    components = []
    seen_nodes = set()
    # Keep track of unique edges encountered in the path
    #unique_edges = set()

    # Start from the nodes of the critical edge
    u, _, _ = critical_edge_key
    start_nodes = [u, v]

    # Attempt to identify paths and cycles within the subgraph, starting only from preferred_starts
    for start_node in start_nodes:
        if start_node in seen_nodes:
            continue  # Skip if the start node has already been seen
        #pdb.set_trace()  # Add this line to start debugging

        stack = [(u, [u], None)]  # Stack to keep track of the traversal
        while stack:
            current_node, current_path, last_color = stack.pop()

            # Skip if current_node is not in subG ????
            if current_node not in subG:
                continue

            #print(f"Processing node: {current_node}, Current path: {current_path}, Last color: {last_color}")

            # Check if the current_node has been seen before in the current path
            if current_node in current_path[:-1]:
                print(f"Cycle detected (back to a previously visited node): {current_path}")
                components.append(('cycle', current_path))
                break

            # and current_node != start_node
            # Check if current_node leads back to one of the start nodes, forming a cycle
            if current_node in start_nodes and len(current_path) > 2:
                cycle_path = current_path  # + [current_node]
                components.append(('cycle', cycle_path))
                print(f"Cycle detected: {cycle_path}")
                break

            seen_nodes.add(current_node)

            # Flag to check if the current node leads to further valid paths
            leads_to_valid_path = False

            for neighbor in subG[current_node]:
                # Avoid revisiting the immediate parent in the path
                if len(current_path) > 1 and neighbor == current_path[-2]:
                    continue

                # # Check if the neighbor is already in the path (exclude the start node)
                # if neighbor in current_path and neighbor != start_node:
                #     continue

                # Check the color of the edge to the neighbor
                edge_color = subG[current_node][neighbor][0]['color']  # Assuming each edge has a 'color' attribute

                # Check if the edge color alternates from the last one and is one of the allowed colors
                if edge_color in [a, b] and edge_color != last_color:
                    new_path = current_path + [neighbor]
                    stack.append((neighbor, new_path, edge_color))
                    leads_to_valid_path = True

            # If the current node does not lead to any valid path, check if it's a valid path end
            if not leads_to_valid_path and len(current_path) >= 3:
                # Check if all vertices in the path are unique (excluding the start node)
                unique_vertices = set(current_path[1:])  # Exclude the start node
                if len(unique_vertices) == len(current_path) - 1:
                    components.append(('path', current_path))
                    #print(f"Path found: {current_path}")
                    interchange_colors(edge_manager, a, b, components)
                    return components, adj_matrix  # Debatable, if I don't return then it will search for more paths

                else:
                    components.append(('cycle', current_path))
                    print(f"Cycle found: {current_path}")


        print(f"Finished processing from start node {start_node}")

    if not components:
        print("No paths or cycles found in the subgraph.")

    print("Updated Subgraph Matrix:")
    #print_matrix(adj_matrix)

    return components, adj_matrix


def interchange_cycle_colors(edge_manager, color_a, color_b, components):
    print("=== Before Interchange ===")
    #print("edge_to_color_dict:", edge_manager.edge_to_color_dict)
    #print("color_to_edge_dict:", edge_manager.color_to_edge_dict)
    #print("path", components)
    counter = 0
    for component in components:
        type_, cycle = component

        if type_ != 'cycle':
            continue  # Skip non-path components

        for i in range(len(cycle) - 1):
            u, v = cycle[i], cycle[i + 1]

            if u == cycle[0]:
                counter += 1
                if counter > 1:
                    break

            # Find the count of the edge with color a, if not found, with color b
            count_a = edge_manager.find_edge_count(u, v, color_a)
            count_b = edge_manager.find_edge_count(u, v, color_b)
            edge_count = count_a if count_a is not None else count_b

            # If neither color a nor color b is found, continue to the next edge
            if edge_count is None:
                print("Problem no color found for edge", u, v)
                continue

            # Determine the edge key to update
            edge_key = (u, v, edge_count) if (u, v, edge_count) in edge_manager.edge_to_color_dict else (v, u, edge_count)
            current_color = edge_manager.edge_to_color_dict[edge_key]
            if current_color == color_a:
                new_color = color_b
            elif current_color == color_b:
                new_color = color_a
            else:
                continue  # If the current color is neither 'color_a' nor 'color_b', skip to the next

            edge_manager.color_edge(u, v, new_color, edge_count)

    print("=== After Interchange ===")
    #print("edge_to_color_dict:", edge_manager.edge_to_color_dict)
    #print("color_to_edge_dict:", edge_manager.color_to_edge_dict)


def interchange_colors(edge_manager, color_a, color_b, components):
    #print("=== Before Interchange ===")
    #print("edge_to_color_dict:", edge_manager.edge_to_color_dict)
    #print("color_to_edge_dict:", edge_manager.color_to_edge_dict)
    #print("path", components)

    for component in components:
        type_, path = component

        if type_ != 'path':
            continue  # Skip non-path components

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]

            # Find the count of the edge with color a, if not found, with color b
            count_a = edge_manager.find_edge_count(u, v, color_a)
            count_b = edge_manager.find_edge_count(u, v, color_b)
            edge_count = count_a if count_a is not None else count_b

            # If neither color a nor color b is found, continue to the next edge
            if edge_count is None:
                print("Problem no color found for edge", u, v)
                continue

            # Determine the edge key to update
            edge_key = (u, v, edge_count) if (u, v, edge_count) in edge_manager.edge_to_color_dict else (v, u, edge_count)
            current_color = edge_manager.edge_to_color_dict[edge_key]
            # for edge_count in range(edge_manager.matrix[u][v]):
            #     edge_key = (u, v, edge_count)
            #     reverse_edge_key = (v, u, edge_count)
            #
            #     if edge_key in edge_manager.edge_to_color_dict:
            #         edge_to_update = edge_key
            #     elif reverse_edge_key in edge_manager.edge_to_color_dict:
            #         edge_to_update = reverse_edge_key
            #     else:
            #         continue  # If neither direction of the edge is found, skip to the next

            # Swap colors if it's one of the two colors we're interchanging
            if current_color == color_a:
                new_color = color_b
            elif current_color == color_b:
                new_color = color_a
            else:
                continue  # If the current color is neither 'color_a' nor 'color_b', skip to the next

            edge_manager.color_edge(u, v, new_color, edge_count)

    print("=== After Interchange ===")
    #print("edge_to_color_dict:", edge_manager.edge_to_color_dict)
    #print("color_to_edge_dict:", edge_manager.color_to_edge_dict)


def critical_path(edge_manager, x, y, count):
    edge_key = (x, y, count)
    edge_color_dict = edge_manager.get_edge_color_dict()
    #print("critical dict", edge_color_dict)


    if edge_key not in edge_color_dict:
        # Check for the reverse edge in case the order was swapped
        edge_key = (y, x, count)
        if edge_key not in edge_color_dict:
            #print(f"Error: Edge {edge_key} not found in edge_to_color_dict.")
            return False, f"Edge {edge_key} not found in edge_to_color_dict."

    #print(f"Working on edge key: {edge_key}")

    x_colors = edge_manager.get_colors(x)
    y_colors = edge_manager.get_colors(y)

    x_unique_color = next(iter(x_colors - y_colors), None)
    y_unique_color = next(iter(y_colors - x_colors), None)

    #print(f"Unique a-b colors: {x_unique_color, y_unique_color}")

    # Extract the colors of all edges to find the used colors
    colors_in_use = {color for color in edge_manager.get_edge_color_dict().values() if color is not None}

    all_colors = set(range(1, 201))
    missing_x_colors = all_colors - x_colors
    missing_y_colors = all_colors - y_colors

    # Find common missing colors from x and y minus colors already in use, case part 1
    common_missing_from_used_colors = (missing_x_colors.intersection(missing_y_colors)).intersection(colors_in_use)
    #print("common_missing_from_used_colors", common_missing_from_used_colors)

    if common_missing_from_used_colors:
        color = common_missing_from_used_colors.pop()
        edge_manager.color_edge(x, y, color, count)
        #print(f"x_colors: {x_colors}, y_colors: {y_colors}")
        #print("colors in use", colors_in_use)
        return True, f"Edge {edge_key} colored with {color}."

    unique_color_pairs = [(a, b) for a in (x_colors - y_colors) for b in (y_colors - x_colors)]
    # Try each unique color pair until success or no more pairs, case 1 part 2
    for x_unique_color, y_unique_color in unique_color_pairs:
        #print(f"Trying unique color pair: ({x_unique_color}, {y_unique_color})")
        # You can directly use x_unique_color and y_unique_color since they are already the unique colors
        #print(f"x_colors: {x_colors}, y_colors: {y_colors}")
        #print(f"x_unique_color: {x_unique_color}, y_unique_color: {y_unique_color}")
        #print("EDW")
        #color_to_edge_dict[edge_key] = None  # So that I can read the edge in ab_subgraph
        components, _ = ab_subgraph(edge_manager, x_unique_color, y_unique_color, edge_key)
        #print("components", components)

        for type_, path in components:
            if type_ == "path":
                if len(path) < 2:
                    print("Error: Path has fewer than 2 vertices.")
                    continue

                start, *middle, end = path

                # If the path starts with x and ends with a vertex missing either 'a' or 'b' color, interchange the colors
                missing_end_colors = set(range(1, 11)) - edge_manager.get_colors(end)
                #print("missing_end_colors", missing_end_colors)
                if start == x and (x_unique_color in missing_end_colors or y_unique_color in edge_manager.get_colors(end)):
                    print(f"Found path: {path}")
                    # print("JOINED IF")
                    # print(end)
                    # print(edge_manager.get_colors(end))
                    # print(x_unique_color)
                    # print(y_unique_color)
                    # interchange_colors(edge_manager, x_unique_color, y_unique_color, components)
                    # Because I make the interchange in ab_subgraph and not here it doesn't read the interchanged path so, it needs reverse color. Change return too?
                    edge_manager.color_edge(x, y, x_unique_color if y_unique_color in edge_manager.get_colors(end) else y_unique_color, count)
                    return True, f"Edge {edge_key} is critical. Colored with {x_unique_color if y_unique_color in edge_manager.get_colors(end) else y_unique_color} after interchange."
                # Case 2, I call function cpath for the critical path I found because case 2 works on Q (the critical) path
                cpath(edge_manager, x_unique_color, y_unique_color, path, edge_key)
                # Case 3 to 5
                procedure_seven(edge_manager, path, x_unique_color, y_unique_color, edge_key)
                procedure_five(edge_manager, path, x_unique_color, y_unique_color, edge_key)
                procedure_three(edge_manager, path, x_unique_color, y_unique_color, edge_key)

                # success = cpath(edge_manager, x_unique_color, y_unique_color, path, edge_key)
                # if success:
                #     # If cpath was successful, return from critical_path immediately
                #     return True, "Coloring successful in cpath"

            # If a cycle is found then check the next pair of unique colors
            elif type_ == "cycle":
                if x_unique_color or y_unique_color in unique_color_pairs:
                    continue

                print("Found a cycle. Exiting without interchange.")
                common_missing_colors = missing_x_colors.intersection(missing_y_colors)
                color = common_missing_colors.pop()
                edge_manager.color_edge(x, y, color, count)
                continue

    # If no common missing color from used colors is found, find it from the general set of missing colors
    common_missing_colors = missing_x_colors.intersection(missing_y_colors)

    #print(f"x_colors: {x_colors}, y_colors: {y_colors}")
    #print("colors in use", colors_in_use)
    #print("all colors", all_colors)
    #print("missing_x_colors", missing_x_colors, "missing_y_colors", missing_y_colors)
    #print(f"Common missing unused colors: {common_missing_colors}")

    if common_missing_colors:
        color = common_missing_colors.pop()
        edge_manager.color_edge(x, y, color, count)
        return True, f"Edge {edge_key} colored with {color}."

    # Fallback coloring method: If the edge is still not colored, use the first available color
    edge_color = edge_color_dict.get(edge_key)
    edge_manager.color_edge(x, y, edge_color, count)
    if edge_color is None:
        # first_available_color = next(iter({1, 2, 3, 4, 5, 6, 7, 8, 9, 10}.difference(x_colors, y_colors)))
        first_available_color = next(iter(set(range(1, 201)).difference(x_colors, y_colors)))
        edge_manager.color_edge(x, y, first_available_color, count)
        print(f"[DEBUG] Used fallback to color edge {edge_key} with color {first_available_color}.")
        return True, f"Edge {edge_key} colored with {first_available_color} using fallback method."
    if edge_manager.get_colors(edge_key) is not None:
        return True, f"Edge {edge_key} colored with {edge_color} using fallback method."
    print(f"Error: Edge {edge_key} wasn't colored using critical path method.")
    return False, f"Edge {edge_key} wasn't colored using critical path method."


def cpath(edge_manager, a, b, path, critical_edge_key):
    # A and B may not be needed because of the path
    #print("Joined cpath")
    #print(path)

    edge_color_dict = edge_manager.get_edge_color_dict()
    #print(edge_color_dict)
    color_edge_dict = edge_manager.get_color_edge_dict()
    #print(color_edge_dict)

    all_colors = set(range(1, 11))
    (x, y, edge_key_count) = critical_edge_key

    # Extract the colors of all edges to find the used colors
    colors_in_use = {color for color in edge_manager.get_edge_color_dict().values() if color is not None}

    # Find common missing colors from x and y minus colors already in use, case 2 part 1
    # Check for consecutive u,v vertexes
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]

        u_colors = edge_manager.get_colors(u)
        v_colors = edge_manager.get_colors(v)
        missing_u_colors = all_colors - u_colors
        missing_v_colors = all_colors - v_colors
        common_missing_colors = (missing_u_colors.intersection(missing_v_colors)).intersection(colors_in_use)
        # common_missing_colors = (set(range(1, 11)) - u_colors).intersection(set(range(1, 11)) - v_colors).intersection(colors_in_use)

        # Check for common missing color
        if common_missing_colors:  # it may need check for else
            common_color = common_missing_colors.pop()  # Now common_color is a single color
            # Possible more  efficient way with search only on the edges of the path with the common missing color

            # Attempt to find the count of the edge with color a and if not found, with color b. Edge u,v has either color a or b, so one of them returns None
            count_a = edge_manager.find_edge_count(u, v, a)
            count_b = edge_manager.find_edge_count(u, v, b)
            edge_count = count_a if count_a is not None else count_b

            if edge_count is not None:
                # Recolor the edge (u, v) with common_color
                edge_manager.color_edge(u, v, common_color, edge_count)
                #print(f"Recolored edge ({u}, {v}) with color {common_color}")
                #print(edge_color_dict)
                #critical_path(edge_manager, u, v, count)
                success, message = critical_path(edge_manager, x, y, edge_key_count)
                if success:
                    return True, message  # This will exit from cpath as well
                break  # Exit after recoloring one edge

    # Iterate through each pair of vertices in the path, except consecutive ones
    #print("path1", path)
    for i in range(len(path) - 2):
        for j in range(i + 2, len(path)):
            u, v = path[i], path[j]

            u_colors = edge_manager.get_colors(u)
            v_colors = edge_manager.get_colors(v)
            missing_u_colors = all_colors - u_colors
            missing_v_colors = all_colors - v_colors
            common_missing_colors = (missing_u_colors.intersection(missing_v_colors)).intersection(colors_in_use)

            found_color = False  # Flag to check if a suitable common missing color is found
            for common_color in common_missing_colors:
                # Check if no vertex between u and v (exclusive) is missing color c
                intermediate_missing = False
                for vertex in path[i + 1: j]:  # i+1 till j + 1?
                    if common_color in edge_manager.get_colors(vertex):
                        intermediate_missing = True
                        #print(f"Color {common_color} isn't missing in vertex {vertex} between {u} and {v}")
                        break  # Skip to the next common color

                if not intermediate_missing:
                    found_color = True
                    #print(f"Suitable common missing color {common_color} found between vertices {u} and {v}")
                    # break  # Break out of the loop as a suitable color is found
            # # Check for common missing color
            # if common_missing_colors:
            #     common_color = common_missing_colors.pop()  # C
            #     print(f"Common missing color {common_color} found between vertices {u} and {v}")
            #
            #     # Check if no vertex between u and v is missing color c
            #     no_intermediate_vertex_missing_c = all(common_color not in edge_manager.get_colors(vertex) for vertex in path[i + 1: j])
            #     print("endiameso color", no_intermediate_vertex_missing_c)
            #     if no_intermediate_vertex_missing_c:
                    # Identify the vertex u' that follows u in the path
                    if i + 1 < len(path):  # Check if u' exists
                        u_prime = path[i + 1]
                        #("u'", u_prime)
                        u_prime_colors = edge_manager.get_colors(u_prime)
                        missing_colors_u_prime = colors_in_use.difference(u_prime_colors)
                        # missing_colors_u_prime = set(range(1, 11)) - u_prime_colors.intersection(colors_in_use)

                        if missing_colors_u_prime:
                            missing_color_u_prime = missing_colors_u_prime.pop()  # c' may need different name
                            #print(f"Missing color {missing_color_u_prime} found for vertex {u_prime} after {u}")
                            path = apath(edge_manager, u_prime, missing_color_u_prime, common_color)
                            #print(path)
                            # first_vertex = path[0]
                            # while first_vertex != u and last_vertex != v do
                            # If Apath( u’, c’, c) ends at u then u := u’ else v := u’
                            if path:
                                #print("Path found:", path)
                                components = [('path', path)]
                                interchange_colors(edge_manager, missing_color_u_prime, common_color, components)
                                last_vertex = path[-1]  # Extract the last vertex of the path
                                if last_vertex == u:
                                    u = u_prime
                                    #print("u = u'", u)
                                    #count = edge_manager.find_edge_count(u, v, common_missing_colors)
                                    #critical_path(edge_manager, u, v, count)
                                else:
                                    v = u_prime
                                    #print("v = v'", v)
                                    #count = edge_manager.find_edge_count(u, v, common_missing_colors)
                                    #critical_path(edge_manager, u, v, count)
                            else:
                                print("No path found!!!")
                            #print("Debug")
                            critical_path(edge_manager, x, y, edge_key_count)  # Possible position two tabs in
            if not found_color:
                print(f"No suitable common missing color found between vertices {u} and {v}")
    #return False


# I may need to add a.b.f.g color as parameters in twopath
def twopath(edge_manager, Q, a, b, R, f, g, u, v, c, critical_edge_key):
    """
    Finds a two-colored path starting from 'start_vertex' using 'color_a' and 'color_b',
    and stops if it encounters 'color_c' or cannot continue.
    """

    print("Joined twopath")
    (x, y, edge_key_count) = critical_edge_key

    # Step 1: Find missing colors for endpoints of Q and R
    # q_start, q_end = Q[0], Q[-1]
    # r_start, r_end = R[0], R[-1]

    # # Problem with a,b,f,g how do I get them???
    # a = edge_manager.get_colors(q_start)
    # b = edge_manager.get_colors(q_end)
    # f = edge_manager.get_colors(r_start)
    # g = edge_manager.get_colors(r_end)

    if a != f:
        u_prime = Q[1]  # u'
        v_prime = R[1]  # v'

        # Extract the colors of all edges to find the used colors
        colors_in_use = {color for color in edge_manager.get_edge_color_dict().values() if color is not None}

        u_prime_colors = edge_manager.get_colors(u_prime)
        missing_colors_u_prime = colors_in_use.difference(u_prime_colors)
        missing_color_u_prime = missing_colors_u_prime.pop()  # c'

        v_prime_colors = edge_manager.get_colors(v_prime)
        missing_colors_v_prime = colors_in_use.difference(v_prime_colors)
        missing_color_v_prime = missing_colors_v_prime.pop()  # c''

        if u_prime == v_prime:
            #print("Checking if c' = c''", missing_color_v_prime == missing_color_u_prime)
            missing_color_u_prime = missing_color_v_prime  # Let c' == c''

        #else: kai ola bellow ena tab mesa
        pathA = apath(edge_manager, u_prime, missing_color_u_prime, c)
        pathB = apath(edge_manager, v_prime, missing_color_v_prime, c)
        pathA_end = pathA[-1]
        pathB_end = pathB[-1]
        if (u != u_prime and pathA_end != u) or (v != v_prime and pathB_end != v):
            components = [('path', pathA)]
            interchange_colors(edge_manager, missing_color_u_prime, c, components)
            cpath(edge_manager, a, b, components, critical_edge_key)
        else:
            if u != u_prime:
                components = [('path', pathA)]
                interchange_colors(edge_manager, missing_color_u_prime, c, components)
            elif v != v_prime:
                components = [('path', pathB)]
                interchange_colors(edge_manager, missing_color_v_prime, c, components)

            pathC = apath(edge_manager, x, a, c)
            pathC_end = pathC[-1]
            if pathC_end == v_prime:
                componentsC = [('path', pathC)]
                interchange_colors(edge_manager, a, c, componentsC)

                count_a = edge_manager.find_edge_count(x, u_prime, a)
                count_b = edge_manager.find_edge_count(x, u_prime, b)
                edge_count = count_a if count_a is not None else count_b
                edge_manager.color_edge(x, v_prime, c, edge_count)
                edge_manager.color_edge(x, y, b, edge_key_count)
            else:
                componentsC = [('path', pathC)]
                interchange_colors(edge_manager, a, c, componentsC)

                cpath(edge_manager, f, g, pathC, critical_edge_key)
    else:
        # Interchange the roles of x and y
        # Reverse Q and R, and swap x and y
        #print("Joined twopath else to interchange the roles of x and y")
        #print(Q)
        #print(edge_key_count)
        Q_reversed = Q[::-1]
        R_reversed = R[::-1]
        #print(Q_reversed)
        swapped_critical_edge_key = (y, x, edge_key_count)  # Swapped critical edge key
        #print(swapped_critical_edge_key)

        # Recursive call with the reversed paths and swapped critical edge key
        return twopath(edge_manager, Q_reversed, a, b, R_reversed, f, g, v, u, c, swapped_critical_edge_key)


# I may need to add a.b color as parameters in leave
def leave(edge_manager, H, Q, a, b, critical_edge_key):  # def leave(edge_manager, Q, u, v, c, critical_edge_key):

    colors_in_use = {color for color in edge_manager.get_edge_color_dict().values() if color is not None}
    print(f"Entering leave function with Q path: {Q}")
    (x, y, edge_key_count) = critical_edge_key

    # q_start, q_end = Q[0], Q[-1]
    # a = edge_manager.get_colors(q_start)
    # b = edge_manager.get_colors(q_end)

    # H = nx.Graph()  # Creating an empty subgraph H
    #
    # # Add edges from the path Q to the subgraph H
    # for i in range(len(Q) - 1):
    #     q, w = Q[i], Q[i + 1]
    #     count_a = edge_manager.find_edge_count(q, w, a)
    #     #count_b = edge_manager.find_edge_count(q, w, b)
    #     edge_color = a if count_a is not None else b
    #     H.add_edge(q, w, color=edge_color)
    #
    # print("H after Q", H)
    #
    # # Add other colored edges connected to vertices in Q, depending on the graph it could be faster to search the edges and not the colors.
    # for color, edges in edge_manager.color_to_edge_dict.items():
    #     if color is not None:
    #         for edge in edges:  # Not a good way to fill the rest edges in H, it doesn't take into account the vertices of Q
    #             q, w, _ = edge
    #             if q in Q and w in Q:
    #                 H.add_edge(q, w, color=color)

    #print(edge_manager.get_edge_color_dict())
    #print("H with the rest edges", H)

    # Have to find c first, first we will check for colors that aren't missing at any edge of H, and then we will search for at least three edges that only one of their vertexes is in H
    # Initialize colors_at_all_vertices with the colors of the first vertex in Q
    colors_at_all_vertices = set(edge_manager.get_colors(Q[0]))

    # Intersect the colors of each vertex in Q with colors_at_all_vertices
    for vertex in Q[1:]:
        colors_at_all_vertices &= set(edge_manager.get_colors(vertex))

    # Mapping each color to count of vertices in Q missing this color
    missing_color_counts = {color: 0 for color in colors_in_use}
    for vertex in Q:
        vertex_colors = edge_manager.get_colors(vertex)
        for color in colors_in_use:
            if color not in vertex_colors:
                missing_color_counts[color] += 1

    flag = False
    # Checking edges leaving Q and their colors
    for color in colors_at_all_vertices:
        for edge in edge_manager.color_to_edge_dict.get(color, []):
            u, u_prime = edge[:2]
            #print("BHKA", edge)
            if (u in Q) != (u_prime in Q):  # XOR check
                # Check if the color is missing at any vertex in Q
                if missing_color_counts[color] > 0:
                    flag = True
                    #print(f"Edge {edge} leaves Q and its color {color} is missing at some vertices in Q.")
    if not flag:
        # Find a color that is not missing at any vertex of H and has three or more c-edges leaving H
        c_color = None
        c_edges_leaving_list1 = []  # Reset for each color
        c_edges_leaving_list = []
        for color in colors_at_all_vertices:
            #print("checking color ", color)
            #print(" colors_at_all_vertices in leave function ", colors_at_all_vertices)
            c_edges_leaving = 0
            c_edges_leaving_list1.clear()  # Clear the list for the new color
            for edge in edge_manager.color_to_edge_dict.get(color, []):  # Possible much faster way if we have a big graph is to make a dictionary with only edges in H
                u, u_prime = edge[:2]  # Assuming edges are tuples (u, u', count)
                if (u in Q) != (u_prime in Q):  # XOR operation: True if only one vertex is in Q
                    c_edges_leaving += 1
                    c_edges_leaving_list1.append(edge)
            if c_edges_leaving >= 3:
                c_edges_leaving_list = c_edges_leaving_list1
                c_color = color
                #print("C color: ", c_color)
                break

        if c_color is None:
            #print("PROBLEM!!! No suitable color c found that isn't missing at any vertex of H and has three or more c-edges leaving H.")
            return False

        u, u_prime = c_edges_leaving_list[0][:2]

        # Determine which vertex is in Q (H)
        u = u if u in H else u_prime
        u_colors = edge_manager.get_colors(u)
        missing_colors_at_u = colors_in_use - u_colors - {a, b, c_color}
        #print("colors in use", colors_in_use)
        #print("missing colors at u", missing_colors_at_u)

        # Choose any color from the missing colors as f
        f = next(iter(missing_colors_at_u), None)  # Get the first color or None if the set is empty
        if f is None:
            # Handle the case when there is no suitable missing color f
            #print("vertex u ", u, "a", a, "b", b, "c", c_color)
            #print(edge_manager.edge_to_color_dict)
            #print(edge_manager.color_to_edge_dict)
            #print("PROBLEM!!! No suitable missing color f found for vertex", u)
            f = next(iter(set(range(1, 201)).difference(colors_in_use)))  # I use max 40 colors
            return
        else:
            print("Selected missing color f for vertex", u, "is", f)

        pathA = apath(edge_manager, u, f, c_color)
        #print("pathaa", pathA)
        if pathA is None:
            return
        # Checking if u is in H
        u, v = pathA[-1], pathA[-2]
        u = u if u in H else v
        v = v if v not in H else pathA[-1]  # Can't put u because it will take the previous u and may lead to error if u = v

        count_a = edge_manager.find_edge_count(u, v, c_color)
        count_b = edge_manager.find_edge_count(u, v, f)
        edge_count = count_a if count_a is not None else count_b
        edge_key = (u, v, edge_count)
        reverse_edge_key = (v, u, edge_count)

        color = edge_manager.get_colors(edge_key) if edge_manager.get_colors(edge_key) is not None else edge_manager.get_colors(reverse_edge_key)
        if color is not None:
            print(f"The color of the edge is {color}.")
        else:
            print(f"PROBLEM!!! The edge is not found or not colored.")

        if color is c_color:
            print("Final edge of the path is c-edge", edge_key)
        else:  # If the last edge of the path isn't c colored then the one before that is
            u, v = pathA[-3], pathA[-4]
            u = u if u in H else v
            v = v if v not in H else pathA[-3]  # Can't put u because it will take the previous u and may lead to error if u = v
            count_a = edge_manager.find_edge_count(u, v, c_color)
            count_b = edge_manager.find_edge_count(u, v, f)
            edge_count = count_a if count_a is not None else count_b
            edge_key = (u, v, edge_count)
            reverse_edge_key = (v, u, edge_count)
            color = edge_manager.get_colors(edge_key) if edge_manager.get_colors(edge_key) is not None else edge_manager.get_colors(reverse_edge_key)
            #print("Debug", color)

        if u == v:
            #print("joined u == v in leave")
            components = [('path', pathA)]
            interchange_colors(edge_manager, f, c_color, components)
        else:
            v_colors = edge_manager.get_colors(v)
            missing_colors_at_v = colors_in_use - v_colors - {a, b, c_color, f}
            #print("colors in use", colors_in_use)
            #print("missing colors at v", missing_colors_at_v)

            # Choose any color from the missing colors as f
            g = next(iter(missing_colors_at_v), None)  # Get the first color or None if the set is empty
            if g is None:
                # Handle the case when there is no suitable missing color f
                #print("PROBLEM!!! No suitable missing color g found for vertex", u)
                g = next(iter(set(range(1, 201)).difference(colors_in_use)))  # I use max 40 colors
                critical_path(edge_manager, x, y, edge_key_count)
                return
            else:
                print("Selected missing color g for vertex", u, "is", g)
            print("tsime", u, g, f)
            pathB = apath(edge_manager, u, g, f)
            components = [('path', pathA)]
            interchange_colors(edge_manager, f, c_color, components)
            componentsB = [('path', pathB)]
            interchange_colors(edge_manager, g, f, componentsB)
    else:
        # Find a color that is not missing at any vertex of H and has three or more c-edges leaving H
        c_color = None
        c_edges_leaving_list1 = []  # Reset for each color
        c_edges_leaving_list = []
        for color in colors_at_all_vertices:
            #print("checking color ", color)
            #print(" colors_at_all_vertices in leave function at else part", colors_at_all_vertices)
            c_edges_leaving = 0
            c_edges_leaving_list1.clear()  # Clear the list for the new color
            for edge in edge_manager.color_to_edge_dict.get(color, []):  # Possible much faster way if we have a big graph is to make a dictionary with only edges in H
                u, u_prime = edge[:2]  # Assuming edges are tuples (u, u', count)
                if (u in Q) != (u_prime in Q):  # XOR operation: True if only one vertex is in Q
                    c_edges_leaving += 1
                    c_edges_leaving_list1.append(edge)
            if c_edges_leaving >= 3:
                c_edges_leaving_list = c_edges_leaving_list1
                c_color = color
                #print("C color: ", c_color)
                break

        if c_color is None:
            #print("PROBLEM!!! No suitable color c found that isn't missing at any vertex of H and has three or more c-edges leaving H.")
            return False

        u, u_prime = c_edges_leaving_list[-1][:2]

        # Determine which vertex is in Q (H)
        u = u if u in H else u_prime

        # I want the order of the vertices to be x,v,u,y
        if u == y:
            u, u_prime = c_edges_leaving_list[-2][:2]
            # Determine which vertex is in Q (H)
            u = u if u in H else u_prime

        if u != x and u != y:
            # Extract indexes of x and u in the path Q
            x_index = Q.index(x)
            u_index = Q.index(u)

            w = Q[u_index - 1]
            w_index = Q.index(w)

            # Ensure x_index is smaller than u_index for correct slicing
            if x_index > u_index:
                x_index, u_index = u_index, x_index

            # Check if there are vertices between x and u
            if u_index - x_index > 1:
                # There are vertices between x and u, choose a random one
                v_index = random.randint(x_index + 1, u_index - 1)
                v = Q[v_index]
                #print(f"Randomly selected vertex between x and u: {v}")
            else:
                # interchange roles of x and y
                #print("DEBUG: No vertices between x and u.")
                #print(Q)
                #print(edge_key_count)
                Q_reversed = Q[::-1]
                #print(Q_reversed)
                swapped_critical_edge_key = (y, x, edge_key_count)  # Swapped critical edge key
                #print(swapped_critical_edge_key)
                # Recursive call with the reversed path and swapped critical edge key
                return leave(edge_manager, H, Q_reversed, a, b, swapped_critical_edge_key)  # Possible no need for u,v,c

            count_a = edge_manager.find_edge_count(w, v, a)
            count_b = edge_manager.find_edge_count(w, v, b)
            edge_count = count_a if count_a is not None else count_b
            edge_manager.uncolor_edge(edge_manager, w, v, edge_count)  # Possible uncolor of edge from G' but not sure. Also, probably doesn't change anything at least in leave function

            path = Q[x_index:w_index + 1]
            #print(path)
            components = [('path', path)]
            interchange_colors(edge_manager, a, b, components)

            edge_manager.color_edge(x, y, b, edge_key_count)


# I may need to add a.b color as parameters in seven
def procedure_seven(edge_manager, Q, a, b, critical_edge_key):

    print("Joined seven")
    # Check if Q has exactly 7 vertices
    if len(Q) == 7:
        print(f"Q {Q}")
        z1, z2, z3, z4, z5 = Q[1:-1]  # Extract z1 to z5 from Q (excluding x and y)
    else:
        #print("DEBUG!!! Q does not have exactly 7 vertices.")
        #print(Q)
        return

    edge_color_dict = edge_manager.get_edge_color_dict()
    #print(edge_color_dict)
    color_edge_dict = edge_manager.get_color_edge_dict()
    #print(color_edge_dict)

    # region Building H

    colors_in_use = {color for color in edge_manager.get_edge_color_dict().values() if color is not None}
    #print(f"Entering leave function with Q path: {Q}")
    (x, y, edge_key_count) = critical_edge_key

    # q_start, q_end = Q[0], Q[-1]
    # a = edge_manager.get_colors(q_start)
    # b = edge_manager.get_colors(q_end)

    H = nx.Graph()  # Creating an empty subgraph H

    # Add edges from the path Q to the subgraph H
    for i in range(len(Q) - 1):
        q, w = Q[i], Q[i + 1]
        count_a = edge_manager.find_edge_count(q, w, a)
        #count_b = edge_manager.find_edge_count(q, w, b)
        edge_color = a if count_a is not None else b
        H.add_edge(q, w, color=edge_color)

    #print("H after Q ", H)

    # Add other colored edges connected to vertices in Q, depending on the graph it could be faster to search the edges and not the colors.
    for color, edges in edge_manager.color_to_edge_dict.items():
        if color is not None:
            for edge in edges:  # Not a good way to fill the rest edges in H, it doesn't take into account the vertices of Q
                q, w, _ = edge
                if q in Q and w in Q:
                    H.add_edge(q, w, color=color)

    #print(edge_manager.get_edge_color_dict())
    #print("H with the rest edges", H)

    # Have to find c first, first we will check for colors that aren't missing at any edge of H, and then we will search for at least three edges that only one of their vertexes is in H
    # Initialize colors_at_all_vertices with the colors of the first vertex in Q
    colors_at_all_vertices = set(edge_manager.get_colors(Q[0]))

    # Intersect the colors of each vertex in Q with colors_at_all_vertices
    for vertex in Q[1:]:
        colors_at_all_vertices &= set(edge_manager.get_colors(vertex))
    #print("colors_at_all_vertices", colors_at_all_vertices)

    # Mapping each color to count of vertices in Q missing this color
    missing_color_counts = {color: 0 for color in colors_in_use}
    for vertex in Q:
        vertex_colors = edge_manager.get_colors(vertex)
        for color in colors_in_use:
            if color not in vertex_colors:
                missing_color_counts[color] += 1

    flag = False
    # Checking edges leaving Q and their colors
    for color in colors_at_all_vertices:
        for edge in edge_manager.color_to_edge_dict.get(color, []):
            u, u_prime = edge[:2]
            #print("edge in double for", edge, u, u_prime)
            if (u in Q) != (u_prime in Q):  # XOR check
                #print("bhka sto if xor")
                # Check if the color is missing at any vertex in Q
                if missing_color_counts[color] > 0:
                    #print("bhka sto if missing_color_counts")
                    flag = True
                    #print(f"Edge {edge} leaves Q and its color {color} is missing at some vertices in Q.")
    if not flag:
        # Find a color that is not missing at any vertex of H and has three or more c-edges leaving H
        c = None
        c_edges_leaving_list1 = []  # Reset for each color
        c_edges_leaving_list = []
        #print("BHKA STO IF NOT FLAG STHN LEAVE")
        for color in colors_at_all_vertices:
            #print("checking color ", color)
            #print("colors_at_all_vertices in seven", colors_at_all_vertices)
            c_edges_leaving = 0
            c_edges_leaving_list1.clear()  # Clear the list for the new color
            for edge in edge_manager.color_to_edge_dict.get(color, []):  # Possible much faster way if we have a big graph is to make a dictionary with only edges in H
                u, u_prime = edge[:2]  # Assuming edges are tuples (u, u', count)
                if (u in Q) != (u_prime in Q):  # XOR operation: True if only one vertex is in Q
                    c_edges_leaving += 1
                    c_edges_leaving_list1.append(edge)
            if c_edges_leaving >= 3:
                c_edges_leaving_list = c_edges_leaving_list1
                c = color
                #print("C color: ", c)
                break

        if c is None:
            #print("PROBLEM!!! No suitable color c found that isn't missing at any vertex of H and has three or more c-edges leaving H in leave.")
            return False

        u, u_prime = c_edges_leaving_list[0][:2]
        # Determine which vertex is in Q (H)
        u = u if u in H else u_prime

        if u == x:
            #print(" if u == x ", x, u)
            u, u_prime = c_edges_leaving_list[1][:2]
            u = u if u in H else u_prime
            #print(" new u ", x, u)
    else:
        print("PROBLEM!!! with the edges leaving Q and their colors")
    # endregion

    # Assign a new color to the edge if certain conditions are met
    if len(colors_in_use) < ((len(H.edges()) / (len(H.nodes()) - 1)) / 2):
        q = next(iter(set(range(1, 201)).difference(colors_in_use)))  # I use max 40 colors
        edge_manager.color_edge(x, y, q, edge_key_count)
    else:
        # There are two or more edges leaving H that are colored with the same color
        leave(edge_manager, H, Q, a, b, critical_edge_key)

        # c1 = edge_manager.get_colors(x)
        # c = colors_in_use.difference(c1)
        # If there is no cb-critical path
        for vertex in H.nodes():
            R = apath(edge_manager, vertex, c, b)
            if R is not None:
                #print(f"Path found starting from vertex {vertex}: {R}")
                return R
            else:
                #print("No alternating path found in H with colors c and b.")
                critical_path(edge_manager, x, y, edge_key_count)
        missing_color = []
        missing_colors_in_R = {}
        missing_colors_in_Q = {}
        #print(R)
        if R is not None:
            for vertex in R:
                vertex_colors = edge_manager.get_colors(vertex)
                missing_color += colors_in_use.difference(vertex_colors)
                for color in colors_in_use:
                    if color not in vertex_colors:
                        if color not in missing_colors_in_R:
                            missing_colors_in_R[color] = [vertex]
                        else:
                            missing_colors_in_R[color].append(vertex)
        else:
            #print("No alternating path found in H with colors c and b.")
            critical_path(edge_manager, x, y, edge_key_count)
        # Check for duplicates
        missing_color_set = set(missing_color)
        if len(missing_color) != len(missing_color_set):  # There are duplicate colors in missing_color so path R contains at least two vertices of a common missing color.
            cpath(edge_manager, c, b, R, critical_edge_key)
        else:
            #print("the cb-critical path R doesnt contain two vertices of a common missing color")
            # Fill in missing colors for each vertex in Q
            for vertex in Q:
                vertex_colors = edge_manager.get_colors(vertex)
                for color in colors_in_use:
                    if color not in vertex_colors:
                        if color not in missing_colors_in_Q:
                            missing_colors_in_Q[color] = [vertex]
                        else:
                            missing_colors_in_Q[color].append(vertex)

                # Find common missing colors between Q and R
                common_missing_colors = {}
                for color in missing_colors_in_Q:
                    if color in missing_colors_in_R:
                        common_missing_colors[color] = {
                            "Q": missing_colors_in_Q[color],
                            "R": missing_colors_in_R[color]
                        }

                if common_missing_colors:
                    #print("Common missing colors with corresponding vertices in Q and R:")
                    color, vertices_info = common_missing_colors.popitem()
                    vertices_in_Q = vertices_info['Q']
                    vertices_in_R = vertices_info['R']

                    # Assuming you want the first vertex from each list
                    w = vertices_in_Q[0]  # Vertex from Q
                    v = vertices_in_R[0]  # Vertex from R
                    "NEED TO MAKE SURE THAT W DOESNT EXIST IN R AND THAT V DOESNT EXIST  IN Q"

                    #print(f"Common missing color: {color}")
                    #print(f"Vertex from Q (w): {w}")
                    #print(f"Vertex from R (v): {v}")
                    twopath(edge_manager, Q, a, b, R, c, b, w, v, color, critical_edge_key)
                else:
                    flag1 = flag2 = False
                    # region Component
                    component1 = apath_for_component(edge_manager, z2, c, b)
                    #print("First for component", component1)
                    # Analyze the result
                    if component1:
                        if component1[0] == component1[-1] and len(component1) == 2:
                            print("A 2-cycle is found.")
                        else:
                            component2 = apath_for_component(edge_manager, z2, b, c)
                            #print("Second path for component", component2)
                            if component2[0] == component2[-1] and len(component2) == 2:
                                print("A 2-cycle is found.")
                            else:
                                contains_edge_z2_z3 = contains_edge_z4_z5 = False
                                component2.reverse()  # This modifies the list in place
                                #print(component1)
                                component = component1[1:] + component2
                                # Check for the presence of edges (z2, z3) and (z4, z5) in the combined component
                                for i in range(len(component) - 1):
                                    edge = (component[i], component[i + 1])
                                    if edge == (z2, z3) or edge == (z3, z2):
                                        contains_edge_z2_z3 = True
                                    if edge == (z4, z5) or edge == (z5, z4):
                                        contains_edge_z4_z5 = True

                                # Check if exactly one of the edges is in the combined component
                                if contains_edge_z2_z3 != contains_edge_z4_z5 and component != R:  # If I want to check the if both lists contain the same elements, regardless of the order set(component) == set(R).
                                    #print("Combined component contains exactly one of the two b-edges.")
                                    flag1 = True
                                    components = [('path', component)]
                                    interchange_colors(edge_manager, b, c, components)
                                    # If there is no ab-critical path
                                    for vertex1 in H.nodes():
                                        ab_path = apath(edge_manager, vertex1, a, b)
                                        if ab_path:
                                            print(f"Path found starting from vertex {vertex1}: {ab_path}")
                                            cpath(edge_manager, a, b, ab_path, critical_edge_key)
                                        else:
                                            print("No alternating path found in H with colors c and b.")
                                            critical_path(edge_manager, x, y, edge_key_count)
                                else:
                                    print("Combined component does not contain exactly one of the two b-edges or are different from the path R.", component, R)
                    else:
                        #print("No path or cycle is found (empty list returned).")
                        component1 = apath_for_component(edge_manager, z2, a, c)
                        #print("First for component", component1)
                        # Analyze the result
                        if component1:
                            if component1[0] == component1[-1] and len(component1) == 2:
                                print("A 2-cycle is found.")
                            else:
                                component2 = apath_for_component(edge_manager, z2, a, c)
                                print("Second path for component", component2)
                                if component2[0] == component2[-1] and len(component2) == 2:
                                    print("A 2-cycle is found.")
                                else:
                                    component2.reverse()  # This modifies the list in place
                                    print(component1)
                                    component = component1[1:] + component2
                                    # Check for the presence of a_edges in the combined component

                                    # Variable to count the number of edges with color 'a'
                                    a_edge_count = 0

                                    # Iterate over the component and count the edges with color 'a'
                                    for i in range(len(component) - 1):
                                        u, v = component[i], component[i + 1]
                                        count_a = edge_manager.find_edge_count(u, v, a)
                                        edge_key_1 = (u, v, count_a)
                                        edge_key_2 = (v, u, count_a)  # Reverse edge

                                        # Check if the edge has color 'a'
                                        if edge_manager.edge_to_color_dict.get(edge_key_1) == a or edge_manager.edge_to_color_dict.get(edge_key_2) == a:
                                            a_edge_count += 1

                                    # Check if exactly one 'a' edge is found
                                    if a_edge_count == 1:
                                        print("Component contains exactly one a-edge.")
                                        flag2 = True
                                        components = [('path', component)]
                                        interchange_colors(edge_manager, a, c, components)
                                        # If there is no ab-critical path
                                        for vertex1 in H.nodes():
                                            ab_path = apath(edge_manager, vertex1, a, b)
                                            if ab_path:
                                                print(f"Path found starting from vertex {vertex1}: {ab_path}")
                                                cpath(edge_manager, a, b, ab_path, critical_edge_key)
                                            else:
                                                print("No alternating path found in H with colors c and b.")
                                                critical_path(edge_manager, x, y, edge_key_count)
                                    else:
                                        print(f"Component contains {a_edge_count} a-edges.")
                # endregion
                if flag1 and flag2 is False: # Possible inside the above "else"!!!
                    # Case 3.1 and 3.2 here
                    for vertex1 in H:
                        S = apath_or_cycles(edge_manager, vertex1, b, c)
                        if len(S) >= 4 and all(z in S for z in [z2, z3, z4, z5]):
                            print(f"Found bc-alternating path or cycle S containing z2, z3, z4, z5: {S}")
                            # Check the two specified orders
                            indices_z2_z3_z5_z4 = [S.index(z2), S.index(z3), S.index(z5), S.index(z4)] if all(
                                z in S for z in [z2, z3, z5, z4]) else None
                            indices_z3_z2_z4_z5 = [S.index(z3), S.index(z2), S.index(z4), S.index(z5)] if all(
                                z in S for z in [z3, z2, z4, z5]) else None
                            indices_z2_z3_z4_z5 = [S.index(z2), S.index(z3), S.index(z4), S.index(z5)] if all(
                                z in S for z in [z2, z3, z4, z5]) else None
                            indices_z3_z2_z5_z4 = [S.index(z3), S.index(z2), S.index(z5), S.index(z4)] if all(
                                z in S for z in [z3, z2, z5, z4]) else None

                            if indices_z2_z3_z5_z4 == [sorted(indices_z2_z3_z5_z4)[0] + i for i in range(4)] or indices_z3_z2_z4_z5 == [sorted(indices_z3_z2_z4_z5)[0] + i for i in range(4)]:
                                print(f"Found bc-alternating path for case 1")
                                components = [('path', S)]
                                interchange_colors(edge_manager, b, c, components)
                                ac_cycle = apath_or_cycles(edge_manager, z1, a, c)
                                components1 = [('cycle', ac_cycle)]
                                interchange_cycle_colors(edge_manager, a, c, components1)  # Check again interchange_cycle_colors
                                # If there is no ab-critical path
                                for vertex1 in H.nodes():
                                    ab_path = apath(edge_manager, vertex1, a, b)
                                    if ab_path:
                                        print(f"Path found starting from vertex {vertex1}: {ab_path}")
                                        cpath(edge_manager, a, b, ab_path, critical_edge_key)
                                    else:
                                        print("No alternating path found in H with colors c and b.")
                                        critical_path(edge_manager, x, y, edge_key_count)
                            # Case 3.2
                            if indices_z2_z3_z4_z5 == [sorted(indices_z2_z3_z4_z5)[0] + i for i in range(4)] or indices_z3_z2_z5_z4 == [sorted(indices_z3_z2_z5_z4)[0] + i for i in range(4)]:
                                print(f"Found bc-alternating path for case 2")

                                #if there is no &critical path then npath(x, y, c, b)...


# I may need to add a.b color as parameters in five
def procedure_five(edge_manager, Q, a, b, critical_edge_key):
    print("Joined five")
    # Check if Q has exactly 5 vertices
    if len(Q) == 5:
        print(f"Q {Q}")
        z1, z2, z3 = Q[1:-1]  # Extract z1 to z3 from Q (excluding x and y)
    else:
        print("DEBUG!!! Q does not have exactly 5 vertices.")
        print(Q)
        return

    edge_color_dict = edge_manager.get_edge_color_dict()
    print(edge_color_dict)
    color_edge_dict = edge_manager.get_color_edge_dict()
    print(color_edge_dict)

    # region Building H

    colors_in_use = {color for color in edge_manager.get_edge_color_dict().values() if color is not None}
    #print(f"Entering leave function with Q path: {Q}")
    (x, y, edge_key_count) = critical_edge_key

    # q_start, q_end = Q[0], Q[-1]
    # a = edge_manager.get_colors(q_start)
    # b = edge_manager.get_colors(q_end)

    H = nx.Graph()  # Creating an empty subgraph H

    # Add edges from the path Q to the subgraph H
    for i in range(len(Q) - 1):
        q, w = Q[i], Q[i + 1]
        count_a = edge_manager.find_edge_count(q, w, a)
        #count_b = edge_manager.find_edge_count(q, w, b)
        edge_color = a if count_a is not None else b
        H.add_edge(q, w, color=edge_color)

    print("H after Q", H)

    # Add other colored edges connected to vertices in Q, depending on the graph it could be faster to search the edges and not the colors.
    for color, edges in edge_manager.color_to_edge_dict.items():
        if color is not None:
            for edge in edges:  # Not a good way to fill the rest edges in H, it doesn't take into account the vertices of Q
                q, w, _ = edge
                if q in Q and w in Q:
                    H.add_edge(q, w, color=color)

    print(edge_manager.get_edge_color_dict())
    print("H with the rest edges", H)

    # Have to find c first, first we will check for colors that aren't missing at any edge of H, and then we will search for at least three edges that only one of their vertexes is in H
    # Initialize colors_at_all_vertices with the colors of the first vertex in Q
    colors_at_all_vertices = set(edge_manager.get_colors(Q[0]))

    # Intersect the colors of each vertex in Q with colors_at_all_vertices
    for vertex in Q[1:]:
        colors_at_all_vertices &= set(edge_manager.get_colors(vertex))

    # Mapping each color to count of vertices in Q missing this color
    missing_color_counts = {color: 0 for color in colors_in_use}
    for vertex in Q:
        vertex_colors = edge_manager.get_colors(vertex)
        for color in colors_in_use:
            if color not in vertex_colors:
                missing_color_counts[color] += 1

    flag = False
    # Checking edges leaving Q and their colors
    for color in colors_at_all_vertices:
        for edge in edge_manager.color_to_edge_dict.get(color, []):
            u, u_prime = edge[:2]
            if (u in Q) != (u_prime in Q):  # XOR check
                # Check if the color is missing at any vertex in Q
                if missing_color_counts[color] > 0:
                    flag = True
                    print(f"Edge {edge} leaves Q and its color {color} is missing at some vertices in Q.")
    if not flag:
        # Find a color that is not missing at any vertex of H and has three or more c-edges leaving H
        c = None
        c_edges_leaving_list1 = []  # Reset for each color
        c_edges_leaving_list = []
        for color in colors_at_all_vertices:
            print("checking color ", color)
            c_edges_leaving = 0
            c_edges_leaving_list1.clear()  # Clear the list for the new color
            for edge in edge_manager.color_to_edge_dict.get(color, []):  # Possible much faster way if we have a big graph is to make a dictionary with only edges in H
                u, u_prime = edge[:2]  # Assuming edges are tuples (u, u', count)
                if (u in Q) != (u_prime in Q):  # XOR operation: True if only one vertex is in Q
                    c_edges_leaving += 1
                    c_edges_leaving_list1.append(edge)
            if c_edges_leaving >= 3:
                c_edges_leaving_list = c_edges_leaving_list1
                c = color
                print("C color: ", c)
                break

        if c is None:
            #print("PROBLEM!!! No suitable color c found that isn't missing at any vertex of H and has three or more c-edges leaving H.")
            return False

        # u, u_prime = c_edges_leaving_list[0][:2]
        # # Determine which vertex is in Q (H)
        # u = u if u in H else u_prime
        #
        # if u == x:
        #     print(" if u == x ", x, u)
        #     u, u_prime = c_edges_leaving_list[1][:2]
        #     u = u if u in H else u_prime
        #     print(" new u ", x, u)
    else:
        print("PROBLEM!!! with the edges leaving Q and their colors")
    # endregion

    # Assign a new color to the edge if certain conditions are met
    if len(colors_in_use) < ((len(H.edges()) / (len(H.nodes()) - 1)) / 2):
        q = next(iter(set(range(1, 201)).difference(colors_in_use)))  # I use max 40 colors
        edge_manager.color_edge(x, y, q, edge_key_count)
    else:
        # There are two or more edges leaving H that are colored with the same color
        leave(edge_manager, H, Q, a, b, critical_edge_key)
        # If there is no cb-critical path
        for vertex in H.nodes():
            R = apath(edge_manager, vertex, c, b)
            if R is not None:
                #print(f"Path found starting from vertex {vertex}: {R}")
                return R
            else:
                #print("No alternating path found in H with colors c and b.")
                critical_path(edge_manager, x, y, edge_key_count)
        missing_color = []
        missing_colors_in_R = {}
        #missing_colors_in_Q = {}
        if R is not None:
            for vertex in R:
                vertex_colors = edge_manager.get_colors(vertex)
                missing_color += colors_in_use.difference(vertex_colors)
                for color in colors_in_use:
                    if color not in vertex_colors:
                        if color not in missing_colors_in_R:
                            missing_colors_in_R[color] = [vertex]
                        else:
                            missing_colors_in_R[color].append(vertex)
        else:
            #print("No alternating path found in H with colors c and b.")
            critical_path(edge_manager, x, y, edge_key_count)
        # Check for duplicates
        missing_color_set = set(missing_color)
        if len(missing_color) != len(missing_color_set):  # There are duplicate colors in missing_color so path R contains at least two vertices of a common missing color.
            cpath(edge_manager, c, b, R, critical_edge_key)
        else:
            #print("the cb-critical path R doesnt contain two vertices of a common missing color")
            #print("R has at most seven vertices")
            if R is None:
                #print("No alternating path found in H with colors c and b.")
                critical_path(edge_manager, x, y, edge_key_count)
            elif len(R) == 7:
                #print("R path", R)
                procedure_seven(edge_manager, R, a, b, critical_edge_key)
            else:
                if len(R) == 3:
                    for vertex1 in H:
                        S = apath_or_cycles(edge_manager, vertex1, b, c)
                        if len(S) >= 2 and all(z in S for z in [z2, z3]):
                            print(f"Found bc-alternating path or cycle S containing z2, z3 : {S}")
                            components = [('path', S)]
                            interchange_colors(edge_manager, b, c, components)
                            critical_path(edge_manager, x, y, edge_key_count)
                if len(R) == 5:
                    #print(f"Q {R}")
                    counter = 0
                    z1, w2, w3 = R[1:-1]
                    # region Building T = V( Q and R)
                    T = nx.Graph()  # Creating an empty subgraph H

                    # Add edges from the path Q to the subgraph H
                    for i in range(len(Q) - 1):
                        q, w = Q[i], Q[i + 1]
                        count_a = edge_manager.find_edge_count(q, w, a)
                        # count_b = edge_manager.find_edge_count(q, w, b)
                        edge_color = a if count_a is not None else b
                        T.add_edge(q, w, color=edge_color)

                    print("H after Q", T)

                    for i in range(len(R) - 1):
                        q, w = R[i], R[i + 1]
                        count_a = edge_manager.find_edge_count(q, w, a)
                        # count_b = edge_manager.find_edge_count(q, w, b)
                        edge_color = a if count_a is not None else b
                        if color == b and not T.has_edge(q, w):  # Because Q is an ab-path and R a cb-path then only b-edges can be duplicated
                            T.add_edge(q, w, color=edge_color)

                    #print("H after R", R)

                    # Add other colored edges connected to vertices in Q, depending on the graph it could be faster to search the edges and not the colors.
                    for color, edges in edge_manager.color_to_edge_dict.items():
                        if color is not None:
                            for edge in edges:  # Not a good way to fill the rest edges in H, it doesn't take into account the vertices of Q
                                q, w, _ = edge
                                if (q in Q and w in Q) or (q in R and w in R):
                                    if color == b and not T.has_edge(q, w):
                                        T.add_edge(q, w, color=color)

                    #print(edge_manager.get_edge_color_dict())
                    print("H with the rest edges", T)
                    #endregion
                    # https://prnt.sc/EQ9aaTBqd8ev
                    if True:  # it will be else
                        for i in range(len(c_edges_leaving_list) - 1):  # DOUBLE CHECK THISS
                            q, w = c_edges_leaving_list[i][:2]
                            if q == z2 or q == z3 or w == z2 or w == z3:
                                counter += 1
                        if counter == 2:
                            for vertex1 in T:
                                S = apath_or_cycles(edge_manager, vertex1, b, c)
                                if all(z in S for z in [z2, z3]):
                                    print(f"Found bc-alternating path or cycle S containing z2, z3, z4, z5: {S}")
                                    components = [('path', S)]
                                    interchange_colors(edge_manager, b, c, components)
                                    critical_path(edge_manager, x, y, edge_key_count)
                        else:
                            edge = edge_manager.color_to_edge_dict.get(a, [w2, w3])
                            #print(edge)
                            if edge is None:
                                for vertex1 in T:
                                    S = apath_or_cycles(edge_manager, vertex1, a, b)
                                    if all(z in S for z in [w2, w3]):
                                        print(f"Found bc-alternating path or cycle S containing z2, z3, z4, z5: {S}")
                                        components = [('path', S)]
                                        interchange_colors(edge_manager, a, b, components)
                                        critical_path(edge_manager, x, y, edge_key_count)
                            else:
                                if len(colors_in_use) < ((len(T.edges()) / (len(T.nodes()) - 1)) / 2):
                                    q = next(iter(set(range(1, 201)).difference(colors_in_use)))  # I use max 40 colors
                                    edge_manager.color_edge(x, y, q, edge_key_count)
                                else:
                                    # There are two or more edges leaving H that are colored with the same color
                                    "case 4.1 to 4.5"


# I may need to add a.b color as parameters in five
def procedure_three(edge_manager, Q, a, b, critical_edge_key):
    print("Joined three")
    # Check if Q has exactly 3 vertices
    if len(Q) == 3:
        print(f"Q {Q}")
        z = Q[1:-1]  # Extract z from Q (excluding x and y)
    else:
        print("DEBUG!!! Q does not have exactly 3 vertices.")
        print(Q)
        return

    edge_color_dict = edge_manager.get_edge_color_dict()
    #print(edge_color_dict)
    color_edge_dict = edge_manager.get_color_edge_dict()
    #print(color_edge_dict)

    # region Building H

    colors_in_use = {color for color in edge_manager.get_edge_color_dict().values() if color is not None}
    #print(f"Entering leave function with Q path: {Q}")
    (x, y, edge_key_count) = critical_edge_key

    # q_start, q_end = Q[0], Q[-1]
    # a = edge_manager.get_colors(q_start)
    # b = edge_manager.get_colors(q_end)

    H = nx.Graph()  # Creating an empty subgraph H

    # Add edges from the path Q to the subgraph H
    for i in range(len(Q) - 1):
        q, w = Q[i], Q[i + 1]
        count_a = edge_manager.find_edge_count(q, w, a)
        #count_b = edge_manager.find_edge_count(q, w, b)
        edge_color = a if count_a is not None else b
        H.add_edge(q, w, color=edge_color)

    print("H after Q", H)

    # Add other colored edges connected to vertices in Q, depending on the graph it could be faster to search the edges and not the colors.
    for color, edges in edge_manager.color_to_edge_dict.items():
        if color is not None:
            for edge in edges:  # Not a good way to fill the rest edges in H, it doesn't take into account the vertices of Q
                q, w, _ = edge
                if q in Q and w in Q:
                    H.add_edge(q, w, color=color)

    #print(edge_manager.get_edge_color_dict())
    print("H with the rest edges", H)

    # Have to find c first, first we will check for colors that aren't missing at any edge of H, and then we will search for at least three edges that only one of their vertexes is in H
    # Initialize colors_at_all_vertices with the colors of the first vertex in Q
    colors_at_all_vertices = set(edge_manager.get_colors(Q[0]))

    # Intersect the colors of each vertex in Q with colors_at_all_vertices
    for vertex in Q[1:]:
        colors_at_all_vertices &= set(edge_manager.get_colors(vertex))

    # Mapping each color to count of vertices in Q missing this color
    missing_color_counts = {color: 0 for color in colors_in_use}
    for vertex in Q:
        vertex_colors = edge_manager.get_colors(vertex)
        for color in colors_in_use:
            if color not in vertex_colors:
                missing_color_counts[color] += 1

    flag = False
    # Checking edges leaving Q and their colors
    for color in colors_at_all_vertices:
        for edge in edge_manager.color_to_edge_dict.get(color, []):
            u, u_prime = edge[:2]
            if (u in Q) != (u_prime in Q):  # XOR check
                # Check if the color is missing at any vertex in Q
                if missing_color_counts[color] > 0:
                    flag = True
                    print(f"Edge {edge} leaves Q and its color {color} is missing at some vertices in Q.")
    if not flag:
        # Find a color that is not missing at any vertex of H and has three or more c-edges leaving H
        c = None
        c_edges_leaving_list1 = []  # Reset for each color
        c_edges_leaving_list = []
        for color in colors_at_all_vertices:
            print("checking color ", color)
            c_edges_leaving = 0
            c_edges_leaving_list1.clear()  # Clear the list for the new color
            for edge in edge_manager.color_to_edge_dict.get(color, []):  # Possible much faster way if we have a big graph is to make a dictionary with only edges in H
                u, u_prime = edge[:2]  # Assuming edges are tuples (u, u', count)
                if (u in Q) != (u_prime in Q):  # XOR operation: True if only one vertex is in Q
                    c_edges_leaving += 1
                    c_edges_leaving_list1.append(edge)
            if c_edges_leaving >= 3:
                c_edges_leaving_list = c_edges_leaving_list1
                c = color
                print("C color: ", c)
                break

        if c is None:
            #print("PROBLEM!!! No suitable color c found that isn't missing at any vertex of H and has three or more c-edges leaving H.")
            return False

        # u, u_prime = c_edges_leaving_list[0][:2]
        # # Determine which vertex is in Q (H)
        # u = u if u in H else u_prime
        #
        # if u == x:
        #     print(" if u == x ", x, u)
        #     u, u_prime = c_edges_leaving_list[1][:2]
        #     u = u if u in H else u_prime
        #     print(" new u ", x, u)
    else:
        print("PROBLEM!!! with the edges leaving Q and their colors")
    # endregion

    # Assign a new color to the edge if certain conditions are met
    if len(colors_in_use) < ((len(H.edges()) / (len(H.nodes()) - 1)) / 2):
        q = next(iter(set(range(1, 201)).difference(colors_in_use)))  # I use max 40 colors
        edge_manager.color_edge(x, y, q, edge_key_count)
    else:
        # There are two or more edges leaving H that are colored with the same color
        leave(edge_manager, H, Q, a, b, critical_edge_key)
        for vertex1 in H:
            S = apath_or_cycles(edge_manager, vertex1, a, c)
            if all(z1 in S for z1 in [y, z]):
                print(f"Found bc-alternating path or cycle S containing z and y {S}")
                components = [('path', S)]
                interchange_colors(edge_manager, a, c, components)
                critical_path(edge_manager, x, y, edge_key_count)


"""VISUALISATION"""


def visualize_graph(matrix, edge_colors):
    G = nx.MultiGraph()  # MultiGraph since we are dealing with multi-graphs

    # Convert integer colors to actual color strings
    # color_map = {
    #     1: "red",
    #     2: "blue",
    #     3: "green",
    #     4: "yellow",
    #     5: "purple",
    #     6: "orange",
    #     7: "pink",
    #     8: "cyan",
    #     9: "magenta",
    #     10: "black",
    #     11: "brown",
    #     12: "teal",
    #     13: "olive",
    #     14: "navy",
    #     15: "maroon",
    #     16: "lime",
    #     17: "aqua",
    #     18: "silver",
    #     19: "gold",
    #     20: "indigo",
    #     21: "violet",
    #     22: "crimson",
    #     23: "coral",
    #     24: "darkgreen",
    #     25: "dodgerblue",
    #     26: "firebrick",
    #     27: "fuchsia",
    #     28: "goldenrod",
    #     29: "gray",
    #     30: "hotpink",
    #     31: "ivory",
    #     32: "khaki",
    #     33: "lavender",
    #     34: "lightgreen",
    #     35: "mistyrose",
    #     36: "navajowhite",
    #     37: "orangered",
    #     38: "palegoldenrod",
    #     39: "peru",
    #     40: "rosybrown",
    #     None: "grey"
    # }


    # Adding edges to the graph
    webcolors_names = list(webcolors.CSS3_NAMES_TO_HEX.keys())

    # Ensure the number of colors needed doesn't exceed the available colors
    num_colors = min(200, len(webcolors_names))

    # Convert color names to a dictionary with integer keys
    color_map = {i + 1: webcolors_names[i] for i in range(num_colors)}

    for u in range(len(matrix)):
        for v in range(u, len(matrix)):
            for edge_count in range(matrix[u][v]):
                int_color = edge_colors.get((u, v, edge_count), 10)  # default color is black
                G.add_edge(u, v, color=color_map[int_color])

    pos = nx.spring_layout(G)
    plt.figure(figsize=(10, 7))

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500)
    nx.draw_networkx_labels(G, pos)

    # Draw edges manually to handle multi-graphs
    for u, v, key, data in G.edges(keys=True, data=True):
        rad = 0.3 * key
        e = FancyArrowPatch(pos[u], pos[v], arrowstyle='-|>', connectionstyle=f'arc3,rad={rad}', mutation_scale=10.0, lw=2, alpha=0.5, color=data['color'])
        plt.gca().add_patch(e)

    plt.axis('off')
    plt.show()


def visualize_subgraph(components, edge_colors, matrix):
    G = nx.Graph()

    for type_, path in components:
        if type_ == 'complex':
            continue  # If it's complex, we currently skip.

        for i in range(len(path) - 1):
            # Iterate over all possible edge counts since we don't have that info in 'components'
            color = 'black'  # Default color
            for edge_count in range(matrix[path[i]][path[i + 1]]):  # matrix should be accessible here
                edge = (path[i], path[i + 1], edge_count)
                reverse_edge = (path[i + 1], path[i], edge_count)

                if edge in edge_colors:
                    color = edge_colors[edge]
                    break
                elif reverse_edge in edge_colors:
                    color = edge_colors[reverse_edge]
                    edge = reverse_edge  # Use the reverse edge if that's the one in edge_colors
                    break

            if edge not in G.edges():
                G.add_edge(edge[0], edge[1], color=color)
            #print(f"Added edge {edge[:2]} with color {color}")  # Debug output

    edge_col_list = [G[u][v]['color'] for u, v in G.edges()]

    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color="skyblue", node_size=1000, width=2.0, alpha=0.6, edge_color=edge_col_list)
    plt.show()


def main():
    filename = 'input.txt'
    edges = read_edges_from_file(filename)
    matrix = create_adjacency_matrix(edges)
    #print("Matrix:")
    #print_matrix(matrix)

    # Create a new subgraph with only the vertices.
    subG = nx.MultiGraph()
    n = len(matrix)
    for i in range(n):
        subG.add_node(i)

    # Create an edge manager for the original graph.
    edge_manager = EdgeManager(matrix)

    # Add edges to the subgraph one by one and color them.
    for edge in edges:
        u, v = edge
        occurrence = subG.number_of_edges(u, v)
        subG.add_edge(u, v)
        #print(f"Attempting to add and color edge {edge}...")

        sub_matrix = create_adjacency_matrix(subG.edges())
        #print("Created adjacency matrix main:")
        #print_matrix(sub_matrix)
        # edge_dict = create_edge_dict_from_matrix(sub_matrix)
        # print("Created edge dictionary:")
        #
        # for k, v in edge_dict.items():
        #     print(f"{k}: {v}")

        #print("Attempting critical_path edge coloring...")
        u, v = edge[0] - 1, edge[1] - 1

        critical_success, message = critical_path(edge_manager, u, v, occurrence)
        if critical_success:
            print(message)
        else:
            print(f"Failed to color edge {edge} using critical_path method.")
            break
        # # Visualize the graph after each edge coloring attempt
        # edge_dict = edge_manager.get_edge_color_dict()
        # visualize_graph(matrix, edge_dict)  # Show the graph with the latest edge colors


    edge_dict = edge_manager.get_edge_color_dict()
    color_edge_dict = edge_manager.get_color_edge_dict()
    print("Updated edge dictionary:", edge_dict)
    print("Updated color dictionary:", color_edge_dict)
    visualize_graph(matrix, edge_dict)  # visualise dont work for more than 4000 edges most likely dude to the library
    if color_edge_dict:  # Check if the dictionary is not empty
        max_key = max(color_edge_dict.keys())
        print("The amount of color that was used is:", max_key)
    else:
        print("The dictionary is empty.")

    """First Try"""
    # edge_dict = create_edge_dict_from_matrix(matrix)
    # print("Initial edge dictionary:", edge_dict)
    #
    # edge_manager = EdgeManager(matrix)
    # uncolored_edges = edge_manager.get_uncolored_edges()
    # print("Uncolored Edges:", uncolored_edges)
    # # print("Initial edge dictionary:", edge_manager.get_edge_dict())
    # # # Delete an edge
    # # edge_manager.delete_edge(1, 2, 0)
    # #
    # # print("Updated edge dictionary:", edge_manager.get_edge_dict())
    # # print("Updated matrix:")
    # # print_matrix(edge_manager.get_matrix())
    #
    # num_colors = 10
    # print("Edge Coloring...")
    # success = dfs_edge_coloring(matrix, num_colors, edge_dict, edge_manager)
    #
    #
    # if success:
    #     for (u, v, edge_count), color in edge_dict.items():
    #         print(f"Edge ({u + 1}, {v + 1}, occurrence {edge_count + 1}) is colored with color {color}")
    #     visualize_graph(matrix, edge_dict)
    #     colored_edges = get_colored_edges(edge_dict)
    #     for edge in colored_edges:
    #         print(edge)  # Format vertex_one, vertex_two, count(0 = 1), color
    # else:
    #     print("No valid edge coloring found for the given number of colors after edge deletion.")

    # test1
    # print("Edge dictionary after coloring:", edge_dict)
    # color_edge_dict_after_deletion = color_to_edges_dict(edge_dict)
    # print("Dictionary by color to edges:", color_edge_dict_after_deletion)

    # test2
    # print("Before deletion:")
    # print("Edge to Color Dictionary:", edge_manager.get_edge_color_dict())
    # print("Color to Edge Dictionary:", edge_manager.get_color_edge_dict())


    # Delete an edge
    """TEST FOR DELETE_EDGE"""
    # edge_manager.delete_edge(0, 2, 0)
    # print("Uncolored Edges:", uncolored_edges)
    # # test2
    # # print("\nAfter deletion:")
    # # print("Edge to Color Dictionary:", edge_manager.get_edge_color_dict())
    # # print("Color to Edge Dictionary:", edge_manager.get_color_edge_dict())
    # print("Matrix:")
    # print_matrix(edge_manager.get_matrix())
    # # test1
    # # Here's the update, we synchronize the dictionaries after the deletion
    # # edge_dict = edge_manager.get_edge_color_dict()
    # # color_edge_dict = edge_manager.get_color_edge_dict()
    # # print("Updated edge dictionary:", edge_dict)
    # # print("Updated color dictionary:", color_edge_dict)
    # # print("Updated matrix:")
    # # print_matrix(edge_manager.get_matrix())
    """TEST FOR SUBGRAPH"""
    # color_map = {
    #     1: "red",
    #     2: "blue",
    #     3: "green",
    #     4: "yellow",
    #     5: "purple",
    #     6: "orange",
    #     7: "pink",
    #     8: "cyan",
    #     9: "magenta",
    #     10: "black"
    # }
    # components, subgraph_adj_matrix = ab_subgraph(edge_manager, 1, 3)
    # for type_, component in components:
    #     if type_ == "complex":
    #         print(f"The component is a {type_} with nodes: {component}")
    #     else:
    #         try:
    #             print(f"The component is a {type_} with nodes: {component.nodes()}")
    #             break
    #         except AttributeError:
    #             print(f"The component is a {type_} with nodes: {component}")
    #
    # # Get the edge-color mapping from the EdgeManager
    # edge_colors_all = edge_manager.get_edge_color_dict()  # Assuming EdgeManager has this method
    #
    # # Filter only for the colors 1 and 3
    # edge_colors_filtered = {edge: color_map[color] for edge, color in edge_colors_all.items() if color in [1, 3]}
    # # edge_colors_filtered = {}
    # # for edge, color in edge_colors_all.items():
    # #     if color is not None and color in [1, 2]:
    # #         edge_colors_filtered[edge] = color_map[color]
    # # print(edge_colors_filtered)
    #
    # # Now visualize the subgraph with the appropriate colors
    # #print(components)
    # #print(edge_colors_filtered)
    # print("Subgraphs' Matrix:")
    # print_matrix(subgraph_adj_matrix)
    #
    # visualize_subgraph(components, edge_colors_filtered, matrix)
    """TEST FOR INTERCHANGE"""
    # interchange_colors(edge_manager, 1, 3)
    # # Update the edge_colors_filtered after interchanging
    # edge_colors_all = edge_manager.get_edge_color_dict()
    # edge_colors_filtered = {edge: color for edge, color in edge_colors_all.items() if color in [1, 3]}
    #
    # visualize_subgraph(components, edge_colors_filtered, matrix)


if __name__ == '__main__':
    main()
