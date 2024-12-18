from find_maximal_subgraphs import *

def pause():
    programPause = input("Press the <ENTER> key to continue...")

with open('./num_nodes.txt') as f:
    num_nodes = [line.rstrip() for line in f]
f.close()

num_nodes = num_nodes[:20]

f = open('../weights/raw/weights_graph_labels.txt', 'w')

maxlen = []

for i in range(1,21):
    dir = '../weights/weights/' + str(i)
    num_n = int(num_nodes[i-1])
    for core in range(num_n+1):
        one_hop_nbrs = find_1hop_neighbors(dir, core)
        one_hop_nbrs = list(one_hop_nbrs)
        maxlen.append(len(one_hop_nbrs))
    
print(max(maxlen))

for i in range(1,21):
    dir = '../weights/weights/' + str(i)
    num_n = int(num_nodes[i-1])
    for core in range(num_n+1):
        one_hop_nbrs = find_1hop_neighbors(dir, core)
        one_hop_nbrs = list(one_hop_nbrs)
        count = 0
        for nbr in one_hop_nbrs:
            count += 1
            if nbr > core:
                with open(dir + '/' + str(i) + '_' + str(core) + '-' + str(nbr) + '.txt') as g:
                    vals = [line.rstrip() for line in g]   
            else:
                with open(dir + '/' + str(i) + '_' + str(nbr) + '-' + str(core) + '.txt') as g:
                    vals = [line.rstrip() for line in g]   
            g.close()
            if count < max(maxlen):
                f.write(vals[1] + ', ')
            else:
                f.write(vals[1] + '\n')     
        # Fill the trailing entries with -1
        while count < 73: #max(maxlen):
            count += 1
            # if count < 6: f.write('-1, ')
            # else: f.write('-1\n')
            if count < 73: f.write('-10, ')#max(maxlen): f.write('0, ')
            else: f.write('-10\n')
f.close()