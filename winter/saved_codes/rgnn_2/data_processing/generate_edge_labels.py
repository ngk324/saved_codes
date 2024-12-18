from find_maximal_subgraphs import *

def pause():
    programPause = input("Press the <ENTER> key to continue...")
with open('./num_nodes.txt') as f:
    num_nodes = [line.rstrip() for line in f]
f.close()

num_nodes = num_nodes[:20]

f = open('../weights/raw/weights_edge_labels.txt', 'w')
for i in range(1,21):
    dir = '../weights/weights/' + str(i)
    num_n = int(num_nodes[i-1])
    for core in range(num_n+1):
        one_hop_nbrs = set([core])
        one_hop_nbrs.update(find_1hop_neighbors(dir, core))
        two_hop_nbrs = find_2hop_neighbors(dir, core)
        two_hop_nbrs.update(one_hop_nbrs)
        two_hop_edges, _ = count_edges_in_subgraph(dir, two_hop_nbrs)
        two_hop_edges = list(two_hop_edges)
        two_hop_edges = sorted(two_hop_edges, key=lambda e: (e[0], e[1]))
        for edge in two_hop_edges:
            edge_start = edge[0]
            edge_end = edge[1]
            if edge_start > edge_end:
                with open(dir + '/' + str(i) + '_' + str(edge_start) + '-' + str(edge_end) + '.txt') as g:
                    vals = [line.rstrip() for line in g]   
            else:
                with open(dir + '/' + str(i) + '_' + str(edge_start) + '-' + str(edge_end) + '.txt') as g:
                    vals = [line.rstrip() for line in g]   
            g.close()
            f.write(str(vals[1])+'\n')
            f.write(str(vals[1])+'\n')
f.close()
