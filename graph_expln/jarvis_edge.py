import numpy as np
from collections import defaultdict
from typing import List, Tuple, Sequence, Optional
import os
import pickle as pk
from jarvis.core.atoms import Atoms

import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def canonize_edge(
    src_id,
    dst_id,
    src_image,
    dst_image,
):
    """Compute canonical edge representation.

    Sort vertex ids
    shift periodic images so the first vertex is in (0,0,0) image
    """
    # store directed edges src_id <= dst_id
    if dst_id < src_id:
        src_id, dst_id = dst_id, src_id
        src_image, dst_image = dst_image, src_image

    # shift periodic images so that src is in (0,0,0) image
    if not np.array_equal(src_image, (0, 0, 0)):
        shift = src_image
        src_image = tuple(np.subtract(src_image, shift))
        dst_image = tuple(np.subtract(dst_image, shift))

    assert src_image == (0, 0, 0)

    return src_id, dst_id, src_image, dst_image

def nearest_neighbor_edges(
    atoms=None,
    cutoff=8,
    max_neighbors=12,
    id=None,
    use_canonize=False,
):
    """Construct k-NN edge list."""
    # returns List[List[Tuple[site, distance, index, image]]]
    all_neighbors = atoms.get_all_neighbors(r=cutoff)
    # print(all_neighbors)

    # if a site has too few neighbors, increase the cutoff radius
    for neighborlist in all_neighbors:
        print(neighborlist)
    min_nbrs = min(len(neighborlist) for neighborlist in all_neighbors)

    attempt = 0
    # print ('cutoff=',all_neighbors)
    if min_nbrs < max_neighbors:
        # print("extending cutoff radius!", attempt, cutoff, id)
        lat = atoms.lattice
        if cutoff < max(lat.a, lat.b, lat.c):
            r_cut = max(lat.a, lat.b, lat.c)
        else:
            r_cut = 2 * cutoff
        attempt += 1

        return nearest_neighbor_edges(
            atoms=atoms,
            use_canonize=use_canonize,
            cutoff=r_cut,
            max_neighbors=max_neighbors,
            id=id,
        )
    # build up edge list
    # NOTE: currently there's no guarantee that this creates undirected graphs
    # An undirected solution would build the full edge list where nodes are
    # keyed by (index, image), and ensure each edge has a complementary edge

    # indeed, JVASP-59628 is an example of a calculation where this produces
    # a graph where one site has no incident edges!

    # build an edge dictionary u -> v
    # so later we can run through the dictionary
    # and remove all pairs of edges
    # so what's left is the odd ones out
    edges = defaultdict(set)
    # print(len(edges))
    for site_idx, neighborlist in enumerate(all_neighbors):
        # sort on distance
        neighborlist = sorted(neighborlist, key=lambda x: x[2])
        distances = np.array([nbr[2] for nbr in neighborlist])
        ids = np.array([nbr[1] for nbr in neighborlist])
        images = np.array([nbr[3] for nbr in neighborlist])

        # find the distance to the k-th nearest neighbor
        max_dist = distances[max_neighbors - 1]
        # max_dist = distances[max_neighbors - 1]

        # keep all edges out to the neighbor shell of the k-th neighbor
        ids = ids[distances <= max_dist]
        images = images[distances <= max_dist]
        distances = distances[distances <= max_dist]

        # keep track of cell-resolved edges
        # to enforce undirected graph construction
        for dst, image in zip(ids, images):
            src_id, dst_id, src_image, dst_image = canonize_edge(
                site_idx, dst, (0, 0, 0), tuple(image)
            )
            if use_canonize:
                edges[(src_id, dst_id)].add(dst_image)
            else:
                edges[(site_idx, dst)].add(tuple(image))
    
    return edges

def draw_graph(atoms):
    cutoff = 4.0  # Example cutoff for neighbors
    neighbors = atoms.get_all_neighbors(r=cutoff)


    # Step 3: Create a graph using networkx
    G = nx.Graph()

    # Add nodes (atoms) to the graph
    for i, coord in enumerate(atoms.cart_coords):
        G.add_node(i, pos=coord, element=atoms.elements[i])

    # Add edges (bonds/neighbors) to the graph
    print('Neighbors')
    print(len(neighbors))
    for i, neighbor_list in enumerate(neighbors):
        print(len(neighbor_list))
        for p,j, dist,img in neighbor_list:
            p_elem = atoms.elements[p]
            j_elem = atoms.elements[j]
            print(p_elem,j_elem)
            if dist <= cutoff:  # Consider neighbors within the cutoff distance
                G.add_edge(i, j, weight=dist)
        print('******************************')

    # Step 4: Plot the 3D graph
    pos = nx.get_node_attributes(G, 'pos')  # Extract positions
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Draw nodes
    node_coords = [pos[i] for i in G.nodes()]
    ax.scatter(*zip(*node_coords), c='r', s=100, label='Atoms')

    # Draw edges (bonds)
    for edge in G.edges():
        node1, node2 = edge
        coord1, coord2 = pos[node1], pos[node2]
        ax.plot([coord1[0], coord2[0]], [coord1[1], coord2[1]], [coord1[2], coord2[2]], 'b-', label='Bonds')

    for i, coord in pos.items():
        label = f'{atoms.elements[i]}'  # Example: Label with element type
        ax.text(coord[0], coord[1], coord[2], label, size=12, zorder=1, color='k')  # Add labels


    plt.show()


with open(os.path.join('../', 'aabbx6_Ef.pickle'), 'rb') as handle:
        dataset = pk.load(handle)

# print(dataset[0])
structure = Atoms.from_dict(dataset[0]['atoms'])
# edge = nearest_neighbor_edges(structure)

# print(edge)
draw_graph(structure)
