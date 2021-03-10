import numpy as np
import matplotlib.pyplot as plt

import os, sys, time
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import random
from ReplayBuffer_v2 import ReplayBuffer
from tensorflow.keras.layers import Input, Lambda, Concatenate
from tensorflow.keras.models import Model

from environment import Environment
from models import MLP, MultiHeadsAttModel, Q_Net, ModelMaker, build_models


t_edge = 0
n_router = 20
n_data = 20

env = Environment(n_router, t_edge, n_data)

neighbors = 4
len_feature = 35
action_space = 4



evolution_interval = K_evo = 10 #Every K epochs, use evolution 
num_models = N_evo = 4 # How many models to train simultaneously

models_data = []
capacity = 200000
TAU = 0.01
alpha = 0.6
GAMMA = 0.98
episode_before_train = 2000
i_episode = 0
mini_batch = 10
loss,score = 0,0
num = 0
total_time = 0
num_param_updates = 0 #how many times we update parameters. After K, we swap to diff model
curr_model = 0 #update data in models_data[curr_model]
f = open('log_router_gqn.txt','w')

l=neighbors
d=128

MLP_in = Input(shape=[len_feature]) #??
Att_in = [Input(shape=(l,d)), Input(shape=(l,d)), Input(shape=(l,d)), Input(shape=(1,l))]
Q_in = [Input(shape=(1,128)), Input(shape=(1,128)), Input(shape=(1,128))]


for i in range(N_evo):
	######build the model#########
	vec = np.zeros((1,neighbors))
	vec[0][0] = 1

	maker = ModelMaker(n_data, len_feature, neighbors, Att_in, action_space, Q_in)
	model, V, relation1, relation2, feature, In, q_net, m1, m2, encoder = maker.make()
	######build the target model#########
	maker_t = ModelMaker(n_data, len_feature, neighbors, Att_in, action_space, Q_in)
	model_t, V_t, relation1_t, relation2_t, feature_t, In_t, q_net_t, m1_t, m2_t, encoder_t = maker_t.make()

	buff=ReplayBuffer(capacity)
	times = [0]*n_data
	#append all data here, unload for each model
	models_data.append(
		[model_t, V_t, relation1_t, relation2_t, feature_t, In_t, q_net_t, m1_t, m2_t, encoder_t, 
		model, V, relation1, relation2, feature, In, q_net, m1, m2, encoder, buff, times ])

def load_data():
	model_t, V_t, relation1_t, relation2_t, feature_t, In_t, q_net_t, m1_t, m2_t, encoder_t, model, V, relation1, relation2, feature, In, q_net, m1, m2, encoder, buff, times = models_data[curr_model]

def store_data():
	models_data[curr_model] = [
		model_t, V_t, relation1_t, relation2_t, feature_t, In_t, q_net_t, m1_t, m2_t, encoder_t, 
		model, V, relation1, relation2, feature, In, q_net, m1, m2, encoder, buff, times ]


load_data() #load 0-th data
#########playing#########
while(1):

	i_episode+=1	
	for i in range(n_data):
		times[i] = times[i] + 1
		if env.data[i].now == env.data[i].target:
			num+=1
			env.data[i].now = np.random.randint(n_router)
			env.data[i].target = np.random.randint(n_router)
			env.data[i].time = 0
			if env.data[i].edge != -1:
				env.edges[env.data[i].edge].load -= env.data[i].size
			env.data[i].size = np.random.rand()
			env.data[i].edge = -1
			total_time+=times[i]
			times[i] = 0

	obs = env.observation()
	adj = env.adjacency()
	ob=[]
	for j in range(n_data):
		ob.append(np.asarray([obs[j]]))
		ob.append(np.asarray([adj[j]]))
	ob.append(np.asarray([vec]))
	action = model.predict(ob)
	act = np.zeros(n_data,dtype = np.int32)
	for j in range(n_data):
		if np.random.rand()<alpha:
			act[j]=random.randrange(action_space)
		else:
			act[j]=np.argmax(action[j])

	reward, done = env.set_action(act)
	next_obs = env.observation()

	buff.add(obs, act, next_obs, reward, done, adj)

	score += sum(reward)
	if i_episode %100 ==0:
		print(int(i_episode/100))
		print(score/100,end='\t')
		f.write(str(score/100)+'\t')
		if num !=0:
			print(total_time/num,end='\t')
			f.write(str(total_time/num)+'\t')
		else :
			print(0,end='\t')
			f.write(str(0)+'\t')
		print(num,end='\t')
		print(loss/100)
		f.write(str(num)+'\t'+str(loss/100)+'\n')
		loss = 0
		score = 0
		num = 0
		total_time = 0
		

	if i_episode < episode_before_train:
		continue
	
	num_param_updates += 1

	#########training#########
	batch = buff.getBatch(mini_batch)
	states,actions,rewards,new_states,dones,adj=[],[],[],[],[],[]
	for i_ in  range(n_data*2+1):
		states.append([])
		new_states.append([])
	for e in batch:
		for j in range(n_data):
			states[j*2].append(e[0][j])
			states[j*2+1].append(e[5][j])
			new_states[j*2].append(e[2][j])
			new_states[j*2+1].append(e[5][j])
		states[n_data*2].append(vec)
		new_states[n_data*2].append(vec)
		actions.append(e[1])
		rewards.append(e[3])
		dones.append(e[4])
		
	actions = np.asarray(actions)
	rewards = np.asarray(rewards)
	dones = np.asarray(dones)
		
	for i_ in  range(n_data*2+1):
		states[i_]=np.asarray(states[i_])
		new_states[i_]=np.asarray(new_states[i_])

	q_values = model.predict(states)
	target_q_values = model_t.predict(new_states)
	for k in range(len(batch)):
		for j in range(n_data):
			if dones[k][j]:
				q_values[j][k][actions[k][j]] = rewards[k][j]
			else:
				q_values[j][k][actions[k][j]] = rewards[k][j] + GAMMA*np.max(target_q_values[j][k])

	history=model.fit(states, q_values, epochs=1, batch_size=10, verbose=0)
	his=0
	for (k,v) in history.history.items():
		his+=v[0]
	loss+=(his/n_data)

	#########training target model#########
	weights = encoder.get_weights()
	target_weights = encoder_t.get_weights()
	for w in range(len(weights)):
		target_weights[w] = TAU * weights[w] + (1 - TAU)* target_weights[w]
	encoder_t.set_weights(target_weights)

	weights = q_net.get_weights()
	target_weights = q_net_t.get_weights()
	for w in range(len(weights)):
		target_weights[w] = TAU * weights[w] + (1 - TAU)* target_weights[w]
	q_net_t.set_weights(target_weights)

	weights = m1.get_weights()
	target_weights = m1_t.get_weights()
	for w in range(len(weights)):
		target_weights[w] = TAU * weights[w] + (1 - TAU)* target_weights[w]
	m1_t.set_weights(target_weights)

	weights = m2.get_weights()
	target_weights = m2_t.get_weights()
	for w in range(len(weights)):
		target_weights[w] = TAU * weights[w] + (1 - TAU)* target_weights[w]
	m2_t.set_weights(target_weights)
	
	#model.save('dgn_2.h5') 

	
	####show####
	'''
	for i in range(n_router):
		plt.scatter(router[i].x, router[i].y, color = 'red')
	for e in edges:
		plt.plot([router[e.start].x,router[e.end].x],[router[e.start].y,router[e.end].y],color='black')
	
	for i in range(n_data):
		if data[i].edge != -1:
			plt.scatter((router[edges[data[i].edge].start].x + router[edges[data[i].edge].end].x)/2, (router[edges[data[i].edge].start].y + router[edges[data[i].edge].end].y)/2, color = 'green')
			plt.text((router[edges[data[i].edge].start].x + router[edges[data[i].edge].end].x)/2, (router[edges[data[i].edge].start].y + router[edges[data[i].edge].end].y)/2, s = str(i),fontsize = 10)
		else :
			plt.scatter(router[data[i].now].x, router[data[i].now].y, color = 'green')
			plt.text(router[data[i].now].x, router[data[i].now].y, s = str(i),fontsize = 10)
	plt.ion()
	plt.pause(20)
	#plt.close()
	'''

	#swap to next model:
	if num_param_updates >= K_evo:
		num_param_updates = 0
		print("swapping to next model! Model {} had loss: {}".format(curr_model, loss))
		# store data 
		store_data()

		#store reward to check which model best (??)

		# check if we're done; then do genetic algo
		if (curr_model+1) >= N_evo:
			#genetic algo here
			curr_model = 0
		else:
			curr_model += 1
		
		# load new data
		load_data()

		# zero out some stuff
		loss = 0
		score = 0
		num = 0
		total_time = 0