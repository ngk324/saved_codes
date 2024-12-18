import numpy as np
import networkx as nx
import os


def to_dual_graph(A, w):
    """
    Given A, which consists of multiple tuples (each representing an edge),
    construct a new graph where each node represents an edge, and there is an
    edge between two nodes if the two edges share a node in the original graph.

    Parameters:
    A (numpy.ndarray): edges of the original graph
    w (numpy.ndarray): edge weights of the original graph

    Returns:
    tuple: containing nodes in the dual graph, node labels, edges of the dual graph,
           edge weights in the dual graph, and group IDs.
    """
   # print("1",len(A))
    # Step 1: Sort all rows of A
    #Ax = np.sort(A, axis=1)
    #Ax = np.unique(Ax, axis=0)
    #Ax = Ax[np.argsort(Ax[:, 0])]
    Ax = A
    #print("2",len(Ax))

    # Step 2: Create nodes from the edges
    n_ = np.arange(0, Ax.shape[0])
  #  print("siize n",len(n_))
    #print(n_)

    # Initialize outputs
    A_ = []
    w_ = []
    #l_ = []

    # Get unique nodes for finding max node
    first_nodes_max = len(np.unique(Ax[:, 0]))
    second_nodes_max = len(np.unique(Ax[:, 1]))
    max_node = max(first_nodes_max, second_nodes_max)
    g_id_ = np.zeros(max_node)

    l_ = np.zeros(len(n_))

    #print("max_node",max_node)
    t = list(Ax[:,0])
    for i in range(len(Ax[:,1])):
        t.append(Ax[i,1])

    for i in range(147):
        if i not in t:
            print("lost",i)
    print(len(np.unique(t)))
    print(len(np.unique(Ax[:,0])))
    print(len(np.unique(Ax[:,1])))
    print(max(np.unique(Ax[:,0])))
    print(max(np.unique(Ax[:,1])))
    
    size =  len(n_) 
    # Step 3: Create edges based on connectivity of original edges
    counter = 0
    for i in range(41,42):#len(n_) - 1):
        #print(i)
        e1 = Ax[i, :]
        for j in range(i + 1, len(n_)):
            e2 = Ax[j, :]
            ediff = e1 - e2

            if ediff[0] == 0 or ediff[1] == 0:
                A_.append((i, j))
                A_.append((j, i ))

                if ediff[0] == 0:
                    w_.append(e1[0])
                    w_.append(e1[0])
                    #l_.append(w[i])
                    #l_.append(w[j])
                    #print(w[i],w[j])
                    #g_id_[i] = e1[0]
                    #g_id_[j] = e1[0]
                else:
                    w_.append(e2[1])
                    w_.append(e2[1])
                    #l_.append(w[i])
                    #l_.append(w[j])
                    #g_id_[i] = e2[1]
                    #g_id_[j] = e2[1]
            #else:
        l_[i] = w[i]
    l_[len(n_)-1] = w[len(n_)-1]
            #counter = counter + 1
    # Convert lists to numpy arrays
    A_ = np.array(A_)
    w_ = np.array(w_)
    l_ = np.array(l_)
    #print(l_)
    return n_, l_, A_, w_, g_id_

with open('./num_nodes.txt') as f:
    num_nodes = [line.rstrip() for line in f]
f.close()

num_nodes = num_nodes[:20]

rawpath = '../weights/raw/'
if not os.path.exists(rawpath):
    os.makedirs(rawpath)

#f = open(rawpath + 'weights_A.txt', 'w')
num_previous_nodes = 0
for i in range(1,2):#21):
    dir = '../weights/raw/subgraphs_' + str(i) +'/'
    dir_edges = '../weights/weights/' + str(i)
    num_n = int(num_nodes[i-1])
    edges_list = []
    for core in range(41,42):#num_n):
        print(i,core)
        with open(dir+str(core) + '_embedding.txt', 'r') as fid:
            A = np.loadtxt(fid, dtype=int, delimiter=',', usecols=(0, 1))
            A = A[::2]

        with open(dir+str(core) + '_embedding_weights.txt', 'r') as fid:
            W = np.loadtxt(fid, dtype=float, usecols=(0))
            W = W[::2]

        # Extract x and y
        x = A[:, 0]
        y = A[:, 1]

        # Initialize adjacency matrix
        adj = np.zeros((A.shape[0], 2), dtype=int)
        adj[:, 0] = A[:, 0]
        adj[:, 1] = A[:, 1]
        
        w_list = list(W)
        #print(len(w_list))

        # Generate dual graph components
        nodes, node_labels, a_mat, w, g_id = to_dual_graph(adj, w_list)

        
       # f = open('../weights/raw/dual_weights_node_labels_'+str(i)+'_'+str(core)+'.txt', 'w')
       # for p in node_labels:
       #     f.write(str(p) + '\n')
       # f.close()

       # f = open('../weights/raw/dual_a_mat_'+str(i)+'_'+str(core)+'.txt', 'w')
       # for p in a_mat:
        #    f.write(str(p[0]) + ',' + str(p[1]) + '\n')
       # f.close()

       # f = open('../weights/raw/dual_edge_labels_'+str(i)+'_'+str(core)+'.txt', 'w')
       # for p in w:
       #     f.write(str(p) + '\n')
       # f.close()

     #   G = nx.Graph()
     #   for edge in A:
     #       G.add_edge(*edge)

     #   dual_G = nx.line_graph(G)