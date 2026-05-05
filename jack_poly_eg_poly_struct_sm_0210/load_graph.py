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

with open('train_poly_struct_poly_eg_2_predictions.pkl', 'rb') as file:
    data_train = pickle.load(file)
with open('test_poly_struct_poly_eg_2_predictions.pkl', 'rb') as file:
    data_test = pickle.load(file)

for i in data_train:
    compound = i
    
    # print(compound)
    org_g = compound+'_org.bin'
    bcast_g = compound+'_bcast.bin'
    if not os.path.isfile('./train_poly_struct_poly_eg_2_graphs/'+org_g):
        continue
    graph_org = dgl.load_graphs('./train_poly_struct_poly_eg_2_graphs/'+org_g)
    graph_bcast = dgl.load_graphs('./train_poly_struct_poly_eg_2_graphs/'+bcast_g)

    # for node in range(graph_bcast.num_nodes()):
    #     node_feature = graph_bcast.ndata['atomic_number'][node]

    #     edges_bcast, edges_org = node_adjacency_matrix(graph_bcast,graph_org, node)
    #     print('Node ',node)
    #     print(edges_bcast)
    #     print(edges_org)

    edges_g1 = get_edges_as_set(graph_org[0][0])
    edges_g2 = get_edges_as_set(graph_bcast[0][0])
    missing_edges = edges_g1 - edges_g2
    # print(f"Missing edges in g2: {missing_edges}")

    missing_bonds = ''
    element_wise_missing_edge = {}
    element_wise_edge = {}
    src, dst = graph_org[0][0].edges()
    src = src.tolist()
    dst = dst.tolist()
    for j in range(0,len(src)):
        p = src[j]
        q = dst[j]
        p_f = graph_org[0][0].ndata['atomic_number'][p]
        q_f = graph_org[0][0].ndata['atomic_number'][q]
        p_element = periodictable.elements[int(p_f)]
        q_element = periodictable.elements[int(q_f)]
        if p_element in element_wise_edge:
            element_wise_edge[p_element] = element_wise_edge[p_element]+1
        else:
            element_wise_edge[p_element] = 1

        if q_element in element_wise_edge:
            element_wise_edge[q_element] = element_wise_edge[q_element]+1
        else:
            element_wise_edge[q_element] = 1
         

    for i in missing_edges:
        a,b = i
        # print(a)
        u_f = graph_org[0][0].ndata['atomic_number'][a]
        v_f = graph_org[0][0].ndata['atomic_number'][b]
        # print(int(u_f))
        u_element = periodictable.elements[int(u_f)]
        v_element = periodictable.elements[int(v_f)]

# Get the element symbol and name
        # print(f"Element: {u_element.name}, Symbol: {u_element.symbol}")
        # print(f"Element: {v_element.name}, Symbol: {v_element.symbol}")
        # print(u_element.symbol+'-'+v_element.symbol)
        #missing_bonds = missing_bonds + u_element.symbol+'-'+v_element.symbol + ' ' 

        if u_element in element_wise_missing_edge:
            element_wise_missing_edge[u_element] = element_wise_missing_edge[u_element]+1
        else:
            element_wise_missing_edge[u_element] = 1

        if v_element in element_wise_missing_edge:
            element_wise_missing_edge[v_element] = element_wise_missing_edge[v_element]+1
        else:
            element_wise_missing_edge[v_element] = 1
    

    for k in element_wise_missing_edge:
        td = []
        missing_percntg = (element_wise_missing_edge[k]/element_wise_edge[k])*100
        td.append(k)
        td.append(missing_percntg)
        td.append('train')
        tr_list.append(td)



#df_test = pd.read_csv('test_ids.csv')
#tr_list = []

for i in data_test:
    compound = i

    # print(compound)
    org_g = compound+'_org.bin'
    bcast_g = compound+'_bcast.bin'
    if not os.path.isfile('./test_poly_struct_poly_eg_2_graphs/'+org_g):
        continue
    graph_org = dgl.load_graphs('./test_poly_struct_poly_eg_2_graphs/'+org_g)
    graph_bcast = dgl.load_graphs('./test_poly_struct_poly_eg_2_graphs/'+bcast_g)

    # for node in range(graph_bcast.num_nodes()):
    #     node_feature = graph_bcast.ndata['atomic_number'][node]

    #     edges_bcast, edges_org = node_adjacency_matrix(graph_bcast,graph_org, node)
    #     print('Node ',node)
    #     print(edges_bcast)
    #     print(edges_org)

    edges_g1 = get_edges_as_set(graph_org[0][0])
    edges_g2 = get_edges_as_set(graph_bcast[0][0])
    missing_edges = edges_g1 - edges_g2
    # print(f"Missing edges in g2: {missing_edges}")
    src, dst = graph_org[0][0].edges()
    src = src.tolist()
    dst = dst.tolist()
    missing_bonds = ''
    element_wise_missing_edge = {}
    element_wise_edge = {}
    for j in range(0,len(src)):
        p = src[j]
        q = dst[j]
        p_f = graph_org[0][0].ndata['atomic_number'][p]
        q_f = graph_org[0][0].ndata['atomic_number'][q]
        p_element = periodictable.elements[int(p_f)]
        q_element = periodictable.elements[int(q_f)]
        if p_element in element_wise_edge:
            element_wise_edge[p_element] = element_wise_edge[p_element]+1
        else:
            element_wise_edge[p_element] = 1

        if q_element in element_wise_edge:
            element_wise_edge[q_element] = element_wise_edge[q_element]+1
        else:
            element_wise_edge[q_element] = 1

    for i in missing_edges:
        a,b = i
        # print(a)
        u_f = graph_org[0][0].ndata['atomic_number'][a]
        v_f = graph_org[0][0].ndata['atomic_number'][b]
        # print(int(u_f))
        u_element = periodictable.elements[int(u_f)]
        v_element = periodictable.elements[int(v_f)]

# Get the element symbol and name
        # print(f"Element: {u_element.name}, Symbol: {u_element.symbol}")
        # print(f"Element: {v_element.name}, Symbol: {v_element.symbol}")
        # print(u_element.symbol+'-'+v_element.symbol)
        #missing_bonds = missing_bonds + u_element.symbol+'-'+v_element.symbol + ' '

        if u_element in element_wise_missing_edge:
            element_wise_missing_edge[u_element] = element_wise_missing_edge[u_element]+1
        else:
            element_wise_missing_edge[u_element] = 1

        if v_element in element_wise_missing_edge:
            element_wise_missing_edge[v_element] = element_wise_missing_edge[v_element]+1
        else:
            element_wise_missing_edge[v_element] = 1


    for k in element_wise_missing_edge:
        td = []
        missing_percntg = (element_wise_missing_edge[k]/element_wise_edge[k])*100
        td.append(k)
        td.append(missing_percntg)
        td.append('test')
        tr_list.append(td)

        



    # print('---------------')


    # g_org = graph_org[0]
    # g_bcast = graph_bcast[0]

    # u1,v1 = g_org.edges()
    # u2,v2 = g_bcast.edges()

    # print(len(graph_org[0][0].edges()[0]))
    # print(len(graph_bcast[0][0].edges()[0]))


    # td.append(compound)
    # td.append(len(graph_org[0][0].edges()[0]))
    # td.append(len(graph_bcast[0][0].edges()[0]))
    # td.append(missing_bonds)
    # tr_list.append(td)
# graph_org = dgl.load_graphs('./perov_graphs_hf/Ag2TiHgCl6_org.bin')
# graph_bcast = dgl.load_graphs('./perov_graphs_hf/Ag2TiHgCl6_bcast.bin')

# Access the loaded graph
# g_org = graph_org[0]
# g_bcast = graph_bcast[0]

# u1,v1 = g_org.edges()
# u2,v2 = g_bcast.edges()

# print(len(graph_org[0][0].edges()[0]))
# print(len(graph_bcast[0][0].edges()[0]))

# print(u2)


# Verify the node features are loaded
# print(g_loaded.ndata['feat'])
#print(element_wise_missing_edge)
header = ['element', 'miss_percentage', 'sample']
with open('poly_struct_poly_eg_sm_0210_element_wise_miss.csv', 'w') as f:
     write = csv.writer(f)
     write.writerow(header)
     write.writerows(tr_list)

