# used for networkx version 1.11 and python 2.7
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import math
import pickle



# global config
# random / all : randomly init for each sample / init all 2^N_Node states
Init_Way = 'random'
Reinit_Time = 1500

# Config of network topology
N_Node = 10
Average_Degree = 2
G_Type = 'nkautomata'
Ws_Nei = 4
Ws_P = 0.9

# Config of Dyn:table/prob
DYN_Type = 'table'
DYN = 'half_rand_automata'
# Desginage_Dyn = {0:0,1:0,2:0,3:0}


# config of way to detect inherent character of this network
Goal_Data_Num = 1024
Draw_Grow_Step = 40

Derrida_Re_Time = 100
# if dyn type is table , the goal data num means we have to explore that much different state
# elif dyn type is prob, means we have to let net spread through that much state and dont require they are different
Draw_Hm_Time = 20


# mark
mark = random.randint(0,100000)
print('random mark'+str(mark))

# store folder
Store_Folder = './bn/'



# generate the network
def generate_network(g_type='random',n_node=5,average_degree = 3):
	# generate random network
	if g_type == 'random':
		dg = nx.DiGraph();
		# add nodes
		for i in range(n_node):
			dg.add_node(i,value = random.randint(0,1))
		# num of edges
		edge_num = n_node * average_degree;
		edges = []
		while len(edges) < edge_num:
			start_node = random.randint(0,n_node-1);
			end_node = random.randint(0,n_node-1);
			if start_node != end_node and [start_node,end_node] not in edges:
				edges.append([start_node,end_node])
		# add those num
		for edge in edges:
			dg.add_edge(edge[0],edge[1]);
		return dg
	# generate n-k automata
	elif g_type == 'nkautomata':
		dg = nx.DiGraph();
		# add nodes with a random value of 0 or 1
		for i in range(N_Node):
			dg.add_node(i,value = random.randint(0,1))
		# add edge : every node with k edge directing to it
		for i in range(N_Node):
			edges_to_this_node = []
			for j in range(Average_Degree):
				# choose a starter and direct to this node
				add = False
				while not add:
					starter = random.randint(0,n_node-1)
					if starter not in edges_to_this_node:
						edges_to_this_node.append(starter)
						dg.add_edge(starter,i)
						add = True
		return dg
	# generate ba scale free network
	elif g_type == 'ba':
		BA = nx.random_graphs.barabasi_albert_graph(N_Node, 2)
		return BA
	elif g_type == 'ws':
		WS = nx.random_graphs.watts_strogatz_graph(N_Node, Ws_Nei, Ws_P)
		return WS


# choose a dyn function table for each node
# return : {0:{0:0,1:1,2:0},1:...}
def func_table(dg,table_type):
	# totally random
	if table_type == 'total_random':
		adj = nx.adjacency_matrix(dg).toarray()
		innodes = adj.sum(axis = 0)
		table = {}
		# table for each node
		for i in range(len(adj)):
			# node table
			node_table = {}
			for j in range(int(math.pow(2,innodes[i]))):
				node_table[j] = random.randint(0,1)
			table[i] = node_table
		return table

	# half random: all node share a unique rule table which is randomly generated
	if table_type == 'half_random':
		# a rule table for all, if one node has only few neighbor , it takes part of the rule table as its own rule
		rule_table = {}
		for i in range(int(math.pow(2,N_Node))):
			rule_table[i] = random.randint(0,1)
		# set rule for each node
		adj = nx.adjacency_matrix(dg).toarray()
		innodes = adj.sum(axis = 0)
		table = {}
		for i in range(N_Node):
			node_table = {}
			for j in range(int(math.pow(2,innodes[i]))):
				node_table[j] = rule_table[j]
			table[i] = node_table
		return table

	# table is minority principle
	if table_type == 'minority':
		adj = nx.adjacency_matrix(dg).toarray()
		innodes = adj.sum(axis = 0)
		table = {}
		print(innodes)
		# table for each node
		for i in range(len(adj)):
			# table for one node
			node_table = {}
			for j in range(int(math.pow(2,innodes[i]))):
				if np.sum(np.array(ten2bin(j,N_Node))) <= innodes[i]/2:
					node_table[j] = 0
				else:
					node_table[j] = 1
			table[i] = node_table
		return table

	# table randomly for NK automata 
	if table_type == 'half_rand_automata':
		table = {}
		every_node_table = {}
		for i in range(int(math.pow(2,Average_Degree))):
			every_node_table[i] = random.randint(0,1)
		for i in range(N_Node):
			table[i] = every_node_table
		return table


# init node data randomly
def init_node(dg):
	for i in range(dg.number_of_nodes()):
		dg.node[i]['value'] = random.randint(0,1)
# init node with a perticular number
def init_node_num(dg,num):
	series = ten2bin(num,N_Node)
	for i in range(N_Node):
		dg.node[i]['value'] = series[i]


# get the innode of each node 
# return:{0:[1,2,3],1:[0,4]...}
def get_innode(adj):
	innodes = {}
	for i in range(adj.shape[0]):
		innode = []
		for j in range(adj.shape[0]):
			if adj[j][i] == 1:
				innode.append(j)
		innodes[i] = innode
	return innodes


# let the net spreading
def spread(dg,table,step = -1,ignore_attractor = 'mind_attractor'):
	node_num = dg.number_of_nodes()
	# data to be returned 
	data = []
	# add initial value to data
	origin_val = []
	for i in range(node_num):
		origin_val.append(dg.node[i]['value'])
	data.append(origin_val)

	# control the circulates
	fall_attractor = False
	run=0
	# step is -1 means no step limitation
	if step == -1:
		step = 10000000000
	while ~fall_attractor and run < step:
		run += 1
		# each step
		next_val = []
		for i in range(node_num):
			# get value of it's neighbors
			b2t = 0
			for iter,val in enumerate(innodes[i]):
				b2t += dg.node[val]['value'] * math.pow(2,len(innodes[i])-iter-1)
			next_val.append(table[i][b2t])
		# print(next_val)
		# set value to the net
		for i in range(node_num):
			dg.node[i]['value'] = next_val[i]

		# if decide to ignore attractor,just continue and never mind if it goes into an attractor
		if ignore_attractor == 'ignore_attractor':
			data.append(next_val)
			continue
		elif ignore_attractor == 'mind_attractor':
			# otherwise we need to check wheter it goes into an attractor
			if next_val not in data:
				data.append(next_val)
			else:
				fall_attractor = True
				# print("the net has into attractor in step:"+str(step))
				break
	return np.array(data)


# let the net spread by probility
def spread_prob(dg,DYN,step = 100):
	node_num = dg.number_of_nodes()
	# data to be returned 
	data = []
	# add initial value to data
	origin_val = []
	for i in range(node_num):
		origin_val.append(dg.node[i]['value'])
	data.append(origin_val)

	# control the circulates
	run=0
	# step is the only limitation because there is no conception like attractor and so on...
	while run < step:
		run += 1
		# each step
		next_val = []
		# if DYN is voter
		if DYN == 'voter':
			for i in range(node_num):
				# num for neighbors who vote for agree
				k = 0.
				# num for all neighbors
				m = len(innodes[i])
				for iter,val in enumerate(innodes[i]):
					if dg.node[val]['value'] == 1:
						k += 1.
				if random.random() < k / m:
					next_val.append(1)
				else:
					next_val.append(0)

		# print(next_val)
		# set value to the net
		for i in range(node_num):
			dg.node[i]['value'] = next_val[i]

		# just add to data to record
		data.append(next_val)
	return np.array(data)


# get hamming distance
def hamming_distance(arr1,arr2):
	return float(np.sum(np.absolute(arr1 - arr2))) / float(arr1.shape[0])

# init the net by some certain order,for instance:000 or 010
def init_orderly(dg,series):
	for i,val in enumerate(series):
		dg.node[i]['value'] = val	


# ten2bin
def ten2bin(ten,len_num):
	# bin value
	series = bin(ten)[2:]
	# bin value in perticular lenth
	bin_series = []
	for i in range(len_num):
		if i - (len_num-len(series)) <0:
			bin_series.append(0)
		else:
			bin_series.append(int(series[i-(len_num-len(series))]))
	return bin_series

# bin2ten
def bin2ten(bin):
	series = bin.tolist()
	for i in range(len(series)):
		series[i] = str(series[i])
	return int(''.join(series),2)



# draw all possible state in a graph
def draw_state_graph(dg,table):
	# first, generate the node of all states
	p_graph = nx.DiGraph()
	for i in range(int(math.pow(2,dg.number_of_nodes()))):
		p_graph.add_node(i)
	# then init the dg and spread dg to add edge to phase_graph
	used = []
	for i in range(int(math.pow(2,dg.number_of_nodes()))):
		if i in used:
			continue
		# ten2bin
		series = ten2bin(i,dg.number_of_nodes())
		init_orderly(dg,series)
		data = spread(dg,table)
		# switch to ten
		data_ten = []
		for j in range(len(data)):
			data_ten.append(bin2ten(data[j]))
		# add data to used
		for j in range(len(data)-1):
			if data_ten[j] not in used:
				# add edge
				p_graph.add_edge(data_ten[j],data_ten[j+1])
				# add to used
				used.append(data_ten[j])
	return p_graph

# get next node
def get_next_node(adj,node):
	for j in range(int(math.pow(2,N_Node))):
		if adj[node][j] == 1:
			return j



# details (cycle / point /eden / basin) in a phase graph
def detail_phase_graph(p_graph):
	
	# adj mat for phase graph
	adjp = nx.adjacency_matrix(p_graph).toarray()
	# fix bug that adj[i][i] could not be 1
	for i in range(p_graph.number_of_nodes()):
		have_next = False
		for j in range(p_graph.number_of_nodes()):
			if adjp[i][j] == 1:
				have_next = True
		if have_next:
			continue
		else:
			adjp[i][i] = 1
	print(adjp)

	# initial
	for i in range(p_graph.number_of_nodes()):
		p_graph.node[i]['eden'] = False
		p_graph.node[i]['fixpoint'] = False
		p_graph.node[i]['cycle'] = False
		p_graph.node[i]['basin2point'] = False
		p_graph.node[i]['basin2cycle'] = False


	# detect graden of eden state
	for i in range(p_graph.number_of_nodes()):
		is_eden = True
		for j in range(p_graph.number_of_nodes()):
			# if some node (expect itself) direct to him ,then he is not a eden state
			if adjp[j][i] == 1 and i != j:
				is_eden = False
		if is_eden:
			p_graph.node[i]['eden'] = True

	# detect fix point
	for i in range(p_graph.number_of_nodes()):
		if adjp[i][i] == 1:
			p_graph.node[i]['fixpoint'] = True
	# detect basin and cycle
	for i in range(p_graph.number_of_nodes()):
		stack = []
		go = True
		curnode = i
		stack.append(curnode)
		while go:
			curnode = get_next_node(adjp,curnode)
			# push current node to stack
			stack.append(curnode)
			# if reach a fix point 
			if p_graph.node[curnode]['fixpoint'] == True:
				# each node in the stack is basin to fix point
				for j in stack:
					if j != curnode:
						p_graph.node[j]['basin2point'] = True
					else:
						break
				go = False
			# if reach a node have reached before
			if curnode in stack[:-1]:
				# each node in the stack is basin to cycle
				before = True
				for j in stack:
					# basin to a cycle
					if j != curnode and before:
						p_graph.node[j]['basin2cycle'] = True
					# cycle
					elif (j == curnode or not before) and p_graph.node[j]['fixpoint'] == False:
						p_graph.node[j]['cycle'] = 'cycle'+str(i)
					if j == curnode:
						before = False
				go = False

	# return p_graph




# exam the num of attractor, lenth of cycle and the num of fix point
def exam_phase_graph(p_graph):
	cycle = {}
	num_of_cycle = 0
	mean_length_of_cycle = 0
	num_of_point = 0
	num_of_basin = 0

	for i in range(p_graph.number_of_nodes()):
		if p_graph.node[i]['fixpoint']:
			num_of_point += 1
		if p_graph.node[i]['basin2point'] or p_graph.node[i]['basin2cycle']:
			num_of_basin += 1
		if p_graph.node[i]['cycle'] != False:
			k = p_graph.node[i]['cycle']
			if cycle.has_key(k):
				cycle[k] += 1
			else:
				cycle[k] = 1

	# cycle :{'a':2,'b':1} => average length = 1.5
	if len(cycle) > 0:
		sum_len = 0
		for k in cycle:
			sum_len += cycle[k]
		avg_length = float(sum_len) / float(len(cycle))
	else:
		avg_length = 0.0
	# (num of basin, num of fix point, average length of cycle attractor)
	print("number of basin:"+str(num_of_basin))
	print("number of attractor(cycle and point):"+str(len(cycle)+num_of_point))
	print("number of point:"+str(num_of_point))
	print("average length of cycle:"+str(avg_length))
	return (num_of_basin,len(cycle)+num_of_point,num_of_point,avg_length)


# chaos degree dective method2 : derrida curve
def derrida_curve():
	hm_dis_x = []
	hm_dis_y = []
	for i in range(Derrida_Re_Time):
		print(str(i)+'re initial...')
		# init node
		init_node(dg)
		data = spread(dg,table,step=200)
		for j in range(data.shape[0]-2):
			ht = data[j]
			ht1 = data[j+1]
			ht2 = data[j+2]
			hm_dis_x.append(hamming_distance(ht,ht1))
			hm_dis_y.append(hamming_distance(ht1,ht2))

	plt.xlim((0, 1))
	plt.ylim((0, 1))
	plt.plot([0,1],[0,1])

	plt.scatter(hm_dis_x,hm_dis_y)
	plt.xlabel('hamming distancd of ht')
	plt.ylabel('hamming distancd of ht1')
	plt.savefig(Store_Folder + 'mark-'+str(mark)+'n:'+str(N_Node)+'-k:'+str(Average_Degree)+'.png')


# another method to detect weather the net is chaos: choose different near number of initial state and let them spread
# observe their distance if the net is chaos the distance will expand explonationaly
def distance_grow(dg,time):
	# several distance changes of different kind of initial state
	change_from_far = []
	change_from_near = []

	gate = float(1)/N_Node
	for i in range(time):
		print(str(i)+'th spread')
		# weather to let the initial state close
		if i < time / 2:
			close = True
		else:
			close = False
		# generate initail series
		state0 = []
		state1 = []
		for i in range(N_Node):
			val = random.randint(0,1)
			state0.append(val)
			# if two series are initially close
			if close:
				# same in a big probility
				if random.random() > gate:
					state1.append(val)
				else:
					state1.append(1-val)
			# if they are initially far away
			else:
				# different in big probility
				if random.random() > gate:
					state1.append(1-val)
				else:
					state1.append(val)

		# if two initial series are same , no significance
		if state1 == state0:
			continue
		# init and spread with certain or prob dyn type
		for i in range(N_Node):
			dg.node[i]['value'] = state0[i]
		if DYN_Type == 'table':
			data0 = spread(dg,table,step = 20,ignore_attractor = 'ignore_attractor')
		elif DYN_Type == 'prob':
			data0 = spread_prob(dg,DYN,step=20)
	
		for i in range(N_Node):
			dg.node[i]['value'] = state1[i]
		if DYN_Type == 'table':
			data1 = spread(dg,table,step = 20,ignore_attractor = 'ignore_attractor')
		elif DYN_Type == 'prob':
			data1 = spread_prob(dg,DYN,step = 20)





		# compare the hamming distance
		hamming_change = []
		for i in range(min(len(data0),len(data1),Draw_Hm_Time)):
			hamming_change.append(hamming_distance(data0[i],data1[i]))
		# add to result
		if close:
			change_from_near.append(hamming_change)
		else:
			change_from_far.append(hamming_change)
	# get average distance change
	avg_change_near = []
	avg_change_far = []
	for i in range(Draw_Hm_Time):
		sum_dis = 0
		sum_time = 0
		for j in range(len(change_from_near)):
			try:
				sum_dis = sum_dis + change_from_near[j][i]
				sum_time += 1
			except IndexError:
				continue
		avg_change_near.append(sum_dis / sum_time)
	for i in range(Draw_Hm_Time):
		sum_dis = 0
		sum_time = 0
		for j in range(len(change_from_far)):
			try:
				sum_dis = sum_dis + change_from_far[j][i]
				sum_time += 1
			except IndexError:
				continue
		avg_change_far.append(sum_dis / sum_time)

	# draw the distance change from far
	plt.figure(1)
	plt.subplot(211)
	plt.plot(avg_change_near[:Draw_Grow_Step])
	# draw the distance change
	plt.ylabel('hm dis from near')
	# plt.savefig(Store_Folder + 'mark-'+str(mark)+'-hm_distance_change_from_near-'+'n:'+str(N_Node)+'-k:'+str(Average_Degree)+'.png')
	plt.subplot(212)
	plt.plot(avg_change_far[:Draw_Grow_Step])
	plt.xlabel('time')
	plt.ylabel('hm dis from far')
	plt.savefig(Store_Folder + 'mark-'+str(mark)+'-hm_distance_change-'+'n:'+str(N_Node)+'-k:'+str(Average_Degree)+'.png')





# graph
print(G_Type)
print(N_Node)
print(Average_Degree)
dg = generate_network(g_type=G_Type,n_node=N_Node,average_degree = Average_Degree)



# adj mat
print(dg)
adj = nx.adjacency_matrix(dg).toarray()

# innode of each node
# example:{1:[2,3],2:[0,1]...}
innodes = get_innode(adj);

# dyn table
# example:{0:{0:0,1:1,2:0},1:...}
# table = rand_func_table(dg)

# with some perticular dyn , no need for a func table cause they spread by proberbility
if DYN_Type == 'table':
	table = func_table(dg,DYN)





print("analyze of the graph")

if Init_Way == 'all':
	# exam length of cycle, num of fix point and so on...
	# phase graph
	p_graph = draw_state_graph(dg,table)
	detail_phase_graph(p_graph)
	for i in range(p_graph.number_of_nodes()):
		print(p_graph.node[i])
	analyze_res = exam_phase_graph(p_graph)
elif Init_Way == 'random':
	# if you use random way to init , that means you cant depict every detail of the characteristic of phase graph
	# so generate a derrida curve instead
	# derrida_curve()

	# dont use derrida curve because node num are too less
	distance_grow(dg,1000)

# generate data
all_data = np.array([[-1]])
# has been explored
has_explored = []



# in this way we aim to generate data until final number of all data reachs goal data num\
i = 0
while len(has_explored) < Goal_Data_Num:
	print('how many has we explored')
	# if i % 10 == 0:
	print(len(has_explored))

	# print('initial time-----'+str(i))
	# init node with a perticular num,if net is too big then use random init instead
	if Init_Way == 'random':
		init_node(dg)
	elif Init_Way == 'all':
		init_node_num(dg,i)
	# init state
	init_state = []
	for j in range(N_Node):
		init_state.append(dg.node[j]['value'])
	# if this state has been explored
	if init_state in has_explored:
		continue
	else:
		has_explored.append(init_state)
	# spread
	if DYN_Type == 'table':
		data = spread(dg,table,step=100)
	elif DYN_Type == 'prob':
		data = spread_prob(dg,DYN,step=100)
	# make each [a,b,c,d] to [a,b,b,c,c,d]
	#(2,3)means(2step,3node)

	# if only one point ,that means it is a eden state and a fix point
	if data.shape[0] == 1:
		temp = np.zeros((2,data.shape[1]))
		temp[0] = data
		temp[1] = data
		data = temp
	expand_data = np.zeros((2*data.shape[0]-2,data.shape[1]))
	for j in range(data.shape[0]):
		# add to has explored
		if j < data.shape[0] - 1:
			cur_state = data[j].tolist()
			# if dyn type is table, then we have to make sure weather the state is explored or not
			if DYN_Type == 'table':
				if cur_state not in has_explored:
					has_explored.append(cur_state)
			# dyn type is prob means we just have to record how many state we have visited
			elif DYN_Type == 'prob':
				has_explored.append(cur_state)

		# generate data to use
		if j == 0:
			expand_data[0] = data[0]
		elif j == data.shape[0] - 1:
			expand_data[expand_data.shape[0]-1] = data[j]
		else:
			# j between first and last
			expand_data[2*j-1] = data[j]
			expand_data[2*j] = data[j]
	# print(expand_data)
	# concat data in every step 
	if all_data[0][0] == -1:
		all_data = expand_data
	else:
		all_data = np.concatenate((all_data,expand_data),axis=0)

print(all_data)
print(all_data.shape)
# change the shape from(step,node_num) => (step,node_num,1)
all_data = all_data[:,:,np.newaxis]
print(all_data.shape)

# save the data
# save time series data
serise_address = Store_Folder + 'mark-'+str(mark)+'-series.pickle'
with open(serise_address,'wb') as f:
	pickle.dump(all_data,f)


# save adj mat
adj_address = Store_Folder + 'mark-'+str(mark)+'-adjmat.pickle'
with open(adj_address,'wb') as f:
	pickle.dump(adj,f)



# save andlyze of the data
if Init_Way == 'all':
	info_address = Store_Folder + 'mark-'+str(mark)+'-average_degree'+str(Average_Degree)+'-'+G_Type+'-analyze-step:'+str(all_data.shape[0])+'-node:'+str(all_data.shape[1])+'-dyn:'+DYN+'-num_of_basin'+str(analyze_res[0])+'-num_of_attractor'+str(analyze_res[1])+'-num_of_point'+str(analyze_res[2])+'-ave_cycle_len'+str(analyze_res[3])+'.pickle'
elif Init_Way == 'random':
	info_address = Store_Folder + 'mark-'+str(mark)+'-average_degree'+str(Average_Degree)+'-'+G_Type+'-analyze-step:'+str(all_data.shape[0])+'-node:'+str(all_data.shape[1])+'-dyn:'+DYN+'.pickle'


with open(info_address,'wb') as f:
	# if we have table ,then save it
	if DYN_Type == 'table':
		pickle.dump(table,f)
	elif DYN_Type == 'prob':
		# then we save one word
		pickle.dump('this is '+DYN+' dyn',f)
