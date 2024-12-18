from find_maximal_subgraphs import *

def pause():
    programPause = input("Press the <ENTER> key to continue...")

with open('./num_nodes.txt') as f:
    num_nodes = [line.rstrip() for line in f]
f.close()

f = open('../weights/raw/weights_graph_indicator.txt', 'w')
for i in range(1,21):
    dir = '../weights/weights/' + str(i)
    num_n = int(num_nodes[i-1])
    for core in range(num_n+1):
        one_hop_nbrs = set([core])
        one_hop_nbrs.update(find_1hop_neighbors(dir, core)) 
        two_hop_nbrs = find_2hop_neighbors(dir, core)
        two_hop_nbrs.update(one_hop_nbrs)
        for j in range(len(two_hop_nbrs)):
            f.write(str(i) + '\n')
f.close()