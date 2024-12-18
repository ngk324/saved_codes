from find_maximal_subgraphs import *
import numpy as np
import os 
import random

def pause():
    programPause = input("Press the <ENTER> key to continue...")

with open('./num_nodes.txt') as f:
    num_nodes = [line.rstrip() for line in f]
f.close()

num_nodes = num_nodes[:50]

rawpath = '../weights/raw/'
if not os.path.exists(rawpath):
    os.makedirs(rawpath)

one_hop_num = 16
two_hop_num = 55
num_graphs = 50
#f = open(rawpath + 'weights_A.txt', 'w')
num_previous_nodes = 0
for i in range(1,num_graphs+1):
    #print(i)
    dir_list = '../weights/weights/edges/edges/edges_' + str(i) + '.txt'
    dir_weights = '../weights/weights/edges/edges/edges_weights_' + str(i) + '.txt'

    num_n = int(num_nodes[i-1])
    filepath = rawpath + 'subgraphs_' + str(i) +'/'
    #print(filepath)
    os.makedirs(os.path.dirname(filepath),exist_ok=True)
    for core in range(num_n):

        print("Graph:",i,"Core:",core)
        #print(core)
        f = open(rawpath + 'subgraphs_' + str(i)+'/' + str(core)+'_subgraph.txt', 'w')
        one_hop_nbrs = set([core])
        one_hop_nbrs.update(find_1hop_neighbors(dir_list, core))
        two_hop_nbrs = find_2hop_neighbors(dir_list, core)
        two_hop_nbrs_alone = list(two_hop_nbrs)
        #print("\n1h nbrs",one_hop_nbrs)
        #print("\n2h nbrs", two_hop_nbrs_alone)

        two_hop_nbrs.update(one_hop_nbrs)
        two_hop_edges, _ = count_edges_in_subgraph(dir_list, two_hop_nbrs)
        two_hop_edges = list(two_hop_edges)
        two_hop_edges = sorted(two_hop_edges, key=lambda e: (e[0], e[1]))
        nodemap = {}
        nodes_sorted = list(two_hop_nbrs)
        nodes_sorted.sort()
        for j in range(len(nodes_sorted)):
            nodemap[nodes_sorted[j]] = j
        
        all_nodes = set(range(0,one_hop_num+two_hop_num))
        diff_nodes = all_nodes.symmetric_difference(two_hop_nbrs)
        diff_nodes_list=list(diff_nodes)
        #print(diff_nodes_list)
        #print("\n2h-1h diff nodes", diff_nodes)
        one_hop_nbrs = list(one_hop_nbrs)
        two_hop_nbrs = list(two_hop_nbrs)
        random_node_permutation_1 = np.random.permutation(one_hop_num)
        random_node_permutation_2 = np.random.permutation(two_hop_num)


        for edge in two_hop_edges:
            edge_start = edge[0]#nodemap[edge[0]]
            edge_end = edge[1]#nodemap[edge[1]]
            #edge_start = nodemap[edge[0]]
            #edge_end = nodemap[edge[1]]
            f.write(str(edge_start)+', '+str(edge_end)+'\n')
            f.write(str(edge_end)+', '+str(edge_start)+'\n')
        num_previous_nodes += len(two_hop_nbrs)
        f.close()


        random_node_permutation_1 = list(np.random.permutation(one_hop_num))
        random_node_permutation_2_temp = list(np.random.permutation(two_hop_num))
        random_node_permutation_2 = [x + one_hop_num for x in random_node_permutation_2_temp]
        


        if core in random_node_permutation_1:
            random_node_permutation_1.remove(core)
        if core in random_node_permutation_2:
            random_node_permutation_2.remove(core)

        perm_lis = random_node_permutation_1 + random_node_permutation_2

        f = open(rawpath + 'subgraphs_' + str(i)+'/' + str(core)+'_permutations_subgraph.txt', 'w')

        perm_list = {core:core}

        counter = 0
        for nodes in perm_lis:
            if counter == core:
                counter += 1
            perm_list.update({counter:nodes})
            counter += 1

        #perm_list.update({24:0})
        #for t in range(24):
        #    perm_list.update({t:t+1})


        for node1,node2 in perm_list.items():
            f.write(str(node1)+", " + str(node2) +"\n")
        f.close()

        print(perm_list)
         
        one_two_hop_mapping = {core:perm_list[0]}
        counter = 1

        #print("1h len:",len(one_hop_nbrs))
        #print("2h len:",len(two_hop_nbrs_alone))
        for k in range(one_hop_num):
            if len(one_hop_nbrs) > k:
                if one_hop_nbrs[k] != core:
                    one_two_hop_mapping.update({one_hop_nbrs[k]:perm_list[counter]})
                    counter = counter + 1
            else:
                na_node = diff_nodes.pop()
                #print(na_node)
                one_two_hop_mapping.update({na_node:perm_list[counter]})
                counter = counter + 1


        for k in range(one_hop_num,two_hop_num+one_hop_num):
            val = k - one_hop_num
            if len(two_hop_nbrs_alone) > val:
                if two_hop_nbrs_alone[val] != core:
                    one_two_hop_mapping.update({two_hop_nbrs_alone[val]:perm_list[counter]})
                    counter = counter + 1
            else:
                na_node = diff_nodes.pop()
                #print(counter,len(perm_list))
                one_two_hop_mapping.update({na_node:perm_list[counter]})
                counter = counter + 1

        one_two_hop_mapping = dict(sorted(one_two_hop_mapping.items()))
        m = open(rawpath + 'subgraphs_' + str(i)+'/' + str(core)+'_embedding_mapping.txt', 'w')
        for og_node,converted_node in one_two_hop_mapping.items():
            m.write(str(og_node) + ' ' + str(converted_node) + '\n')
        m.close()
        w = open(rawpath + 'subgraphs_' + str(i)+'/' + str(core)+'_embedding_weights.txt', 'w')
        a = open(rawpath + 'subgraphs_' + str(i)+'/' + str(core)+'_embedding.txt', 'w')

        edges_list1 = get_edges(dir_list)
        weights1 = get_weights(dir_weights)

        w1 = []
        a1 = []
        for k in range(one_hop_num):
            true_node = k
            if len(one_hop_nbrs) > k:
                true_node = one_hop_nbrs[k]
            else:
                true_node = diff_nodes_list[k-len(one_hop_nbrs)]

            n1 = one_two_hop_mapping[true_node]
            core1 = one_two_hop_mapping[core]

            edge_exists = False

            if core != true_node:
                for p in range(len(edges_list1)):
                    if((edges_list1[p][0] == core and edges_list1[p][1] == true_node) or (edges_list1[p][1] == core and edges_list1[p][0] == true_node)):
                        #w.write(str(weights1[p])+'\n' + str(weights1[p])+'\n')
                        #a.write(str(n1) + ', ' + str(core1) + '\n' + str(core1) + ', ' + str(n1) + '\n')
                        w1.append(weights1[p])
                        #w.append(weights1[p])
                        a1.append([n1,core1])
                        #a.append([core1,n1])
                        edge_exists = True
                        #print(1,str(n1),str(core1))

                if not edge_exists:
                    #w.write(str(-1)+'\n'+str(-1)+'\n')
                    #a.write(str(core1) + ', ' + str(n1) + '\n' + str(n1) + ', ' + str(core1) + '\n')
                    #w.append(-1)
                    w1.append(-1)
                    #a.append([n1,core1])
                    a1.append([core1,n1])

        # Generate a list of indices
        indices = list(range(len(w1)))

        # Shuffle the indices
        random.shuffle(indices)

        # Reorder both lists based on the shuffled indices
        shuffled_w1 = [w1[l] for l in indices]
        shuffled_a1 = [a1[l] for l in indices]

        for p in range(len(w1)):
            w.write(str(shuffled_w1[p]) + '\n' + str(shuffled_w1[p]) + '\n')
            a.write(str(shuffled_a1[p][0])+ ', ' + str(shuffled_a1[p][1]) + '\n' + str(shuffled_a1[p][1]) + ', ' + str(shuffled_a1[p][0]) + '\n')
                    

        w2 = []
        a2 = []
        #counter = 0
        for j in range(one_hop_num):
            for k in range(j+1,one_hop_num):
                edge_exists = False
                true_node1 = j
                true_node2 = k

                if len(one_hop_nbrs)  > j:
                    true_node1 = one_hop_nbrs[j]
                else:
                    true_node1 = diff_nodes_list[j-len(one_hop_nbrs)]

                if len(one_hop_nbrs)  > k:
                    true_node2 = one_hop_nbrs[k]
                else:
                    true_node2 = diff_nodes_list[k-len(one_hop_nbrs)]

            
                node1 = one_two_hop_mapping[true_node1]
                node2 = one_two_hop_mapping[true_node2]

                if core != true_node1 and core != true_node2:
                    for p in range(len(edges_list1)):
                        if((edges_list1[p][0] == true_node1 and edges_list1[p][1] == true_node2) or (edges_list1[p][1] == true_node1 and edges_list1[p][0] == true_node2)):
                            #w.write(str(weights1[p])+'\n' + str(weights1[p])+'\n')
                            #a.write(str(node1) + ', ' + str(node2) + '\n' + str(node2) + ', ' + str(node1) + '\n')
                            w2.append(weights1[p])
                            #w.append(weights1[p])
                            a2.append([node1,node2])
                            edge_exists = True
                            #print(2,str(weights1[p]))
                    if not edge_exists:
                        #w.write(str(-1)+'\n'+str(-1)+'\n')
                        #a.write(str(node1) + ', ' + str(node2) + '\n' + str(node2) + ', ' + str(node1) + '\n')
                        w2.append(-1)
                        #a.append([n1,core1])
                        a2.append([node1,node2])
                
        # Generate a list of indices
        indices = list(range(len(w2)))

        # Shuffle the indices
        random.shuffle(indices)

        # Reorder both lists based on the shuffled indices
        shuffled_w2 = [w2[l] for l in indices]
        shuffled_a2 = [a2[l] for l in indices]

        for p in range(len(indices)):
            w.write(str(shuffled_w2[p]) + '\n' + str(shuffled_w2[p]) + '\n')
            a.write(str(shuffled_a2[p][0])+ ', ' + str(shuffled_a2[p][1]) + '\n' + str(shuffled_a2[p][1]) + ', ' + str(shuffled_a2[p][0]) + '\n')
        
        w3 = []
        a3 = []
        for j in range(one_hop_num):
            for k in range(two_hop_num):
                edge_exists = False
                true_node1 = j
                true_node2 = k+one_hop_num

                if len(one_hop_nbrs) > j:
                    true_node1 = one_hop_nbrs[j]
                else:
                    true_node1 = diff_nodes_list[j-len(one_hop_nbrs)]

                if len(two_hop_nbrs_alone) > k:
                    true_node2 = two_hop_nbrs_alone[k]
                else:
                    true_node2 = diff_nodes_list[k-len(two_hop_nbrs_alone)+one_hop_num-len(one_hop_nbrs)]

            
                node1 = one_two_hop_mapping[true_node1]
                node2 = one_two_hop_mapping[true_node2]

                if core != true_node1 and core != true_node2:
                    for p in range(len(edges_list1)):
                        if((edges_list1[p][0] == true_node1 and edges_list1[p][1] == true_node2) or (edges_list1[p][0] == true_node2 and edges_list1[p][1] == true_node1)):
                            #w.write(str(weights1[p])+'\n' + str(weights1[p])+'\n')
                            #a.write(str(node1) + ', ' + str(node2) + '\n' + str(node2) + ', ' + str(node1) + '\n')
                            w3.append(weights1[p])
                            #w.append(weights1[p])
                            a3.append([node1,node2])
                            edge_exists = True
                            #print(3,str(weights1[p]))
                    if not edge_exists:
                        #w.write(str(-1)+'\n'+str(-1)+'\n')
                        #a.write(str(node1) + ', ' + str(node2) + '\n' + str(node2) + ', ' + str(node1) + '\n')
                        w3.append(-1)
                        a3.append([node1,node2])

        # Generate a list of indices
        indices = list(range(len(w3)))

        # Shuffle the indices
        random.shuffle(indices)

        # Reorder both lists based on the shuffled indices
        shuffled_w3 = [w3[l] for l in indices]
        shuffled_a3 = [a3[l] for l in indices]

        for p in range(len(indices)):
            w.write(str(shuffled_w3[p]) + '\n' + str(shuffled_w3[p]) + '\n')
            a.write(str(shuffled_a3[p][0])+ ', ' + str(shuffled_a3[p][1]) + '\n' + str(shuffled_a3[p][1]) + ', ' + str(shuffled_a3[p][0]) + '\n')

        w4 = []
        a4 = []
        for j in range(two_hop_num):
            for k in range(j+1,two_hop_num):
                edge_exists = False
                true_node1 = j+one_hop_num
                true_node2 = k+one_hop_num

                if len(two_hop_nbrs_alone) > j:
                    true_node1 = two_hop_nbrs_alone[j]
                else:
                    true_node1 = diff_nodes_list[j-len(two_hop_nbrs_alone)+one_hop_num-len(one_hop_nbrs)]

                if len(two_hop_nbrs_alone) > k:
                    true_node2 = two_hop_nbrs_alone[k]
                else:
                    true_node2 = diff_nodes_list[k-len(two_hop_nbrs_alone)+one_hop_num-len(one_hop_nbrs)]

            
                node1 = one_two_hop_mapping[true_node1]
                node2 = one_two_hop_mapping[true_node2]

                if core != true_node1 and core != true_node2:
                    for p in range(len(edges_list1)):
                        if((edges_list1[p][0] == true_node1 and edges_list1[p][1] == true_node2) or (edges_list1[p][1] == true_node1 and edges_list1[p][0] == true_node2)):
                            #w.write(str(weights1[p])+'\n' + str(weights1[p])+'\n')
                            #a.write(str(node1) + ', ' + str(node2) + '\n' + str(node2) + ', ' + str(node1) + '\n')
                            w4.append(weights1[p])
                            #w.append(weights1[p])
                            a4.append([node1,node2])
                            edge_exists = True
                            #print(4,str(weights1[p]))
                    if not edge_exists:
                        #w.write(str(-1)+'\n'+str(-1)+'\n')
                        #a.write(str(node1) + ', ' + str(node2) + '\n' + str(node2) + ', ' + str(node1) + '\n')
                        w4.append(-1)
                        a4.append([node1,node2])

        # Generate a list of indices
        indices = list(range(len(w4)))

        # Shuffle the indices
        random.shuffle(indices)

        # Reorder both lists based on the shuffled indices
        shuffled_w4 = [w4[l] for l in indices]
        shuffled_a4 = [a4[l] for l in indices]

        for p in range(len(indices)):
            w.write(str(shuffled_w4[p]) + '\n' + str(shuffled_w4[p]) + '\n')
            a.write(str(shuffled_a4[p][0])+ ', ' + str(shuffled_a4[p][1]) + '\n' + str(shuffled_a4[p][1]) + ', ' + str(shuffled_a4[p][0]) + '\n')
        
        #break
        w.close()
        a.close()
    #break
#f.close()










