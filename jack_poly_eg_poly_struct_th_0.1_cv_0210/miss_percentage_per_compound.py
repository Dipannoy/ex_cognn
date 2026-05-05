import dgl
import torch
import pandas as pd
import periodictable
import csv
import os
import pickle
# Load the graph from the file

def get_edges_as_set(graph):
    src, dst = graph.edges()
    return set(zip(src.tolist(), dst.tolist()))

def node_adjacency_matrix(g_bcast, g, node_id):
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
    in_bcast_edges = g_bcast.in_edges(node_id, form='uv')  # Incoming edges (edges where node_id is the destination)
    out_bcast_edges = g_bcast.out_edges(node_id, form='uv')  # Outgoing edges (edges where node_id is the source)
    combined_edges_bcast = (torch.cat([in_bcast_edges[0], out_bcast_edges[0]]), torch.cat([in_bcast_edges[1], out_bcast_edges[1]]))

    in_edges = g.in_edges(node_id, form='uv')  # Incoming edges (edges where node_id is the destination)
    out_edges = g.out_edges(node_id, form='uv')  # Outgoing edges (edges where node_id is the source)
    combined_edges = (torch.cat([in_edges[0], out_edges[0]]), torch.cat([in_edges[1], out_edges[1]]))

    return combined_edges_bcast, combined_edges



#df_train = pd.read_csv('train_ids.csv')
tr_list = []

with open('train_poly_struct_poly_eg_4_predictions.pkl', 'rb') as file:
    data_train = pickle.load(file)
with open('test_poly_struct_poly_eg_4_predictions.pkl', 'rb') as file:
    data_test = pickle.load(file)

for i in data_train:
    compound = i
    
    # print(compound)
    org_g = compound+'_org.bin'
    bcast_g = compound+'_bcast.bin'
    if not os.path.isfile('./train_poly_struct_poly_eg_4_graphs/'+org_g):
        continue
    graph_org = dgl.load_graphs('./train_poly_struct_poly_eg_4_graphs/'+org_g)
    graph_bcast = dgl.load_graphs('./train_poly_struct_poly_eg_4_graphs/'+bcast_g)



    edges_g1 = get_edges_as_set(graph_org[0][0])
    edges_g2 = get_edges_as_set(graph_bcast[0][0])
    missing_edges = edges_g1 - edges_g2


    missing_percntg = (len(missing_edges)/len(edges_g1))*100
    td=[]
    td.append(missing_percntg)
    td.append(compound)
    td.append('train')
    tr_list.append(td)



#df_test = pd.read_csv('test_ids.csv')
#tr_list = []

for i in data_test:
    compound = i

    # print(compound)
    org_g = compound+'_org.bin'
    bcast_g = compound+'_bcast.bin'
    if not os.path.isfile('./test_poly_struct_poly_eg_4_graphs/'+org_g):
        continue
    graph_org = dgl.load_graphs('./test_poly_struct_poly_eg_4_graphs/'+org_g)
    graph_bcast = dgl.load_graphs('./test_poly_struct_poly_eg_4_graphs/'+bcast_g)

    edges_g1 = get_edges_as_set(graph_org[0][0])
    edges_g2 = get_edges_as_set(graph_bcast[0][0])
    missing_edges = edges_g1 - edges_g2
    missing_percntg = (len(missing_edges)/len(edges_g1))*100
    td=[]
    td.append(missing_percntg)
    td.append(compound)
    td.append('test')
    tr_list.append(td)
    

        
header = ['miss_percentage', 'compound', 'sample']
with open('poly_eg_miss_percentage_per_compound_4.csv', 'w') as f:
     write = csv.writer(f)
     write.writerow(header)
     write.writerows(tr_list)


