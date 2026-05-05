import dgl
import torch

# Function to get the subgraph for a node and its adjacency matrix
def node_adjacency_matrix(g, node_id):
    # Get the induced subgraph for the neighbors of the node
    sub_nodes = g.successors(node_id)
    #sub_nodes.append(node_id)
    subgraph = dgl.node_subgraph(g, torch.cat([torch.tensor([node_id]), sub_nodes]))
    src_sub, dst_sub = subgraph.edges()
    print("Source subgraph nodes:", src_sub)
    print("Destination subgraph nodes:", dst_sub)
    print(subgraph.ndata[dgl.NID])  # original node IDs
    print(subgraph.edata[dgl.EID])  # original edge IDs

    
    # Get the adjacency matrix of the subgraph
    adj_matrix = subgraph.adjacency_matrix(transpose=False, ctx=torch.device('cpu'))
    
    return adj_matrix

# Example: Get adjacency matrix for each node
for node in range(g.num_nodes()):
    node_adj_matrix = node_adjacency_matrix(g, node)
    print(f"Adjacency matrix for node {node}:\n", node_adj_matrix.to_dense())
    print('**************************')