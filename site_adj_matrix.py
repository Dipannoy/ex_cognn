import dgl
import torch

# Example: Create a random graph with 5 nodes and 10 edges
#g = dgl.rand_graph(3, 5)

u = torch.tensor([0, 1, 2, 3])  # Source nodes
v = torch.tensor([1, 2, 3, 1])  # Destination nodes
g = dgl.graph((u, v))
g = dgl.remove_self_loop(g)
src, dst = g.edges()

# Print the edges
print("Source nodes:", src)
print("Destination nodes:", dst)
# Function to get the subgraph for a node and its adjacency matrix
def node_adjacency_matrix(g, node_id):
    # Get the induced subgraph for the neighbors of the node
    '''
    sub_nodes = g.successors(node_id)
    #sub_nodes.append(node_id)
    #subgraph = dgl.node_subgraph(g, torch.cat([torch.tensor([node_id]), sub_nodes]))
    subgraph = dgl.node_subgraph(g, torch.tensor([node_id]))
    src_sub, dst_sub = subgraph.edges()
    print("Source subgraph nodes:", src_sub)
    print("Destination subgraph nodes:", dst_sub)
    print(subgraph.ndata[dgl.NID])  # original node IDs
    print(subgraph.edata[dgl.EID])  # original edge IDs
    
    in_subgraph = dgl.graph(g.in_edges(node_id, form='uv'))

    print(in_subgraph)
    '''
    in_edges = g.in_edges(node_id, form='uv')  # Incoming edges (edges where node_id is the destination)
    out_edges = g.out_edges(node_id, form='uv')  # Outgoing edges (edges where node_id is the source)
    print(in_edges)
    print(out_edges)
    # Combine the incoming and outgoing edges
    combined_edges = (torch.cat([in_edges[0], out_edges[0]]), torch.cat([in_edges[1], out_edges[1]]))
    print(combined_edges)

# Create the subgraph from these edges
    subgraph = dgl.graph(combined_edges)
    #print(subgraph.ndata[dgl.NID])  # original node IDs
    #print(subgraph.edata[dgl.EID])  # original edge IDs

    
    # Get the adjacency matrix of the subgraph
    adj_matrix = subgraph.adjacency_matrix(transpose=False, ctx=torch.device('cpu'))
    
    return adj_matrix

# Example: Get adjacency matrix for each node
'''
for node in range(g.num_nodes()):
    node_adj_matrix = node_adjacency_matrix(g, node)
    print(f"Adjacency matrix for node {node}:\n", node_adj_matrix.to_dense())
    print('**************************')
'''
node_adj_matrix = node_adjacency_matrix(g, 1)
node = 1
print(f"Adjacency matrix for node {node}:\n", node_adj_matrix.to_dense())
