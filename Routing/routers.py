import numpy as np
import matplotlib.pyplot as plt

np.random.seed(476)
class Router(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.neighbor = []
        self.edge=[]

class Edge(object):
    def __init__(self, x, y, l):
        self.start = x
        self.end = y
        self.len = int(int(l*10)/2+1)
        self.load = 0

class Data(object):
    def __init__(self, x, y, size, priority):
        self.now = x
        self.target = y
        self.size = size
        self.priority = priority
        self.time = 0
        self.edge = -1
        self.neigh = [priority,-1,-1,-1]

router = []
edges = []
t_edge = 0
n_router = 20 #orig. 20

####build the graph####
for i in range(n_router):
    router.append(Router(np.random.random(),np.random.random()))

for i in range(n_router):

    dis = []
    for j in range(n_router):
        dis.append([(router[j].x - router[i].x)**2 + (router[j].y - router[i].y)**2, j])
    dis.sort(key = lambda x: x[0],reverse = False)

    for j in range(n_router):

        if len(router[i].neighbor) == 3:
            break
        if j == 0 :
            continue

        if len(router[dis[j][1]].neighbor) < 3:

            router[i].neighbor.append(dis[j][1])
            router[dis[j][1]].neighbor.append(i)

            if i<dis[j][1]:
                edges.append(Edge(i,dis[j][1],np.sqrt(dis[j][0])))
                router[i].edge.append(t_edge)
                router[dis[j][1]].edge.append(t_edge)
                t_edge += 1
            else:
                edges.append(Edge(dis[j][1],i,np.sqrt(dis[j][0])))
                router[dis[j][1]].edge.append(t_edge)
                router[i].edge.append(t_edge)
                t_edge += 1

for i in range(n_router):
    plt.scatter(router[i].x, router[i].y, color = 'orange')
for e in edges:
    plt.plot([router[e.start].x,router[e.end].x],[router[e.start].y,router[e.end].y],color='black')

data = []
n_data = 20
for i in range(n_data):
    data.append(Data(np.random.randint(n_router),np.random.randint(n_router),np.random.random(),i))

def observation(router, edges, data, n_router, n_data, t_edge):
    obs = []
    for i in range(n_data):
        ob=[]

        ####meta information####
        ob.append(data[i].now)
        ob.append(data[i].target)
        ob.append(data[i].edge)
        ob.append(data[i].size)
        ob.append(data[i].priority)

        ####edge information####
        for j in router[data[i].now].edge:
            ob.append(j)
            ob.append(edges[j].start)
            ob.append(edges[j].end)
            ob.append(edges[j].len)
            ob.append(edges[j].load)

        ####other datas####
        count =0;
        data[i].neigh = []
        data[i].neigh.append(i)

        for j in range(n_data):
            if j==i:
                continue
            if (data[j].now in router[data[i].now].neighbor)|(data[j].now == data[i].now):
                count+=1
                ob.append(data[j].now)
                ob.append(data[j].target)
                ob.append(data[j].edge)
                ob.append(data[j].size)
                ob.append(data[i].priority)
                data[i].neigh.append(j)

            if count==3:
                break
        for j in range(3-count):
            data[i].neigh.append(-1)
            for k in range(5):
                ob.append(-1) #invalid placeholder

        obs.append(np.array(ob))

    return obs

def set_action(act,edges, data, n_data, t_edge):

    reward = [0]*n_data
    done = [False]*n_data

    for i in range(n_data):
        if data[i].edge != -1:
            data[i].time -= 1
            if data[i].time == 0:
                edges[data[i].edge].load -= data[i].size
                data[i].edge = -1

        elif act[i]==0:
            continue

        else:
            t = router[data[i].now].edge[act[i]-1]
            if edges[t].load + data[i].size >1:
                reward[i] = -0.2
            else:
                data[i].edge = t
                data[i].time = edges[t].len
                edges[t].load += data[i].size

                if edges[t].start == data[i].now:
                    data[i].now = edges[t].end
                else:
                    data[i].now = edges[t].start

        if data[i].now == data[i].target:
            reward[i] = 10
            done[i] = True

    return data, edges, reward, done

import os, sys, time
from keras import backend as K
from keras.optimizers import Adam
import tensorflow as tf
import random
from ReplayBuffer_v2 import ReplayBuffer
from keras.layers import Dense, Dropout, Conv2D, Input, Lambda, Flatten, TimeDistributed, merge
from keras.layers import Add, Reshape, MaxPooling2D, Concatenate, Embedding, RepeatVector
from keras.models import Model
from keras.layers.core import Activation
from keras.utils import np_utils,to_categorical
from keras.engine.topology import Layer

neighbors = 4
len_feature = 35
action_space = 4
def Adjacency(data,n_data):
    adj = []
    for j in range(n_data):
        l = to_categorical(data[j].neigh,num_classes=n_data)
        for i in range(4):
            if data[j].neigh[i] == -1:
                l[i]=np.zeros(n_data)
        adj.append(l)
    return adj

def MLP():

    In_0 = Input(shape=[len_feature])
    h = Dense(128, activation='relu',kernel_initializer='random_normal')(In_0)
    h = Dense(128, activation='relu',kernel_initializer='random_normal')(h)
    h = Reshape((1,128))(h)
    model = Model(input=In_0,output=h)
    return model

def MultiHeadsAttModel(l=2, d=128, dv=16, dout=128, nv = 8 ):

    v1 = Input(shape = (l, d))
    q1 = Input(shape = (l, d))
    k1 = Input(shape = (l, d))
    ve = Input(shape = (1, l))

    v2 = Dense(dv*nv, activation = "relu",kernel_initializer='random_normal')(v1)
    q2 = Dense(dv*nv, activation = "relu",kernel_initializer='random_normal')(q1)
    k2 = Dense(dv*nv, activation = "relu",kernel_initializer='random_normal')(k1)

    v = Reshape((l, nv, dv))(v2)
    q = Reshape((l, nv, dv))(q2)
    k = Reshape((l, nv, dv))(k2)
    v = Lambda(lambda x: K.permute_dimensions(x, (0,2,1,3)))(v)
    k = Lambda(lambda x: K.permute_dimensions(x, (0,2,1,3)))(k)
    q = Lambda(lambda x: K.permute_dimensions(x, (0,2,1,3)))(q)

    att = Lambda(lambda x: K.batch_dot(x[0],x[1] ,axes=[3,3]) / np.sqrt(dv))([q,k])# l, nv, nv
    att = Lambda(lambda x: K.softmax(x))(att)
    out = Lambda(lambda x: K.batch_dot(x[0], x[1],axes=[3,2]))([att, v])
    out = Lambda(lambda x: K.permute_dimensions(x, (0,2,1,3)))(out)

    out = Reshape((l, dv*nv))(out)

    T = Lambda(lambda x: K.batch_dot(x[0],x[1]))([ve,out])

    out = Dense(dout, activation = "relu",kernel_initializer='random_normal')(T)
    model = Model(inputs=[q1,k1,v1,ve], outputs=out)
    return model

def Q_Net(action_dim):

    I1 = Input(shape = (1, 128))
    I2 = Input(shape = (1, 128))
    I3 = Input(shape = (1, 128))

    h1 = Flatten()(I1)
    h2 = Flatten()(I2)
    h3 = Flatten()(I3)

    h = Concatenate()([h1,h2,h3])
    V = Dense(action_dim,kernel_initializer='random_normal')(h)

    model = Model(input=[I1,I2,I3],output=V)
    return model




evolution_interval = K_evo = 500 #Every K epochs, use evolution 
num_models = N_evo = 5 # How many models to train simultaneously
evolution_rate = eta_evo = 0.5
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
f = open('log_router_gqn_same_buf.txt','w')

for i in range(N_evo):
    ######build the model#########
    encoder = MLP()
    m1 = MultiHeadsAttModel(l=neighbors)
    m2 = MultiHeadsAttModel(l=neighbors)
    q_net = Q_Net(action_dim = action_space)
    vec = np.zeros((1,neighbors))
    vec[0][0] = 1

    In= []
    for j in range(n_data):
        In.append(Input(shape=[len_feature]))
        In.append(Input(shape=(neighbors,n_data)))
    In.append(Input(shape=(1,neighbors)))
    feature = []
    for j in range(n_data):
        feature.append(encoder(In[j*2]))

    feature_ = Concatenate(axis=1)(feature)

    relation1 = []
    for j in range(n_data):
        T = Lambda(lambda x: K.batch_dot(x[0],x[1]))([In[j*2+1],feature_])
        relation1.append(m1([T,T,T,In[n_data*2]]))

    relation1_ = Concatenate(axis=1)(relation1)

    relation2 = []
    for j in range(n_data):
        T = Lambda(lambda x: K.batch_dot(x[0],x[1]))([In[j*2+1],relation1_])
        relation2.append(m2([T,T,T,In[n_data*2]]))

    V = []
    for j in range(n_data):
        V.append(q_net([feature[j],relation1[j],relation2[j]]))

    model = Model(input=In,output=V) #Model.load_model("dgn.h5")
    model.compile(optimizer=Adam(lr = 0.001), loss='mse')

    ######build the target model#########
    encoder_t = MLP()
    m1_t = MultiHeadsAttModel(l=neighbors)
    m2_t = MultiHeadsAttModel(l=neighbors)
    q_net_t = Q_Net(action_dim = action_space)
    In_t= []
    for j in range(n_data):
        In_t.append(Input(shape=[len_feature]))
        In_t.append(Input(shape=(neighbors,n_data)))
    In_t.append(Input(shape=(1,neighbors)))

    feature_t = []
    for j in range(n_data):
        feature_t.append(encoder_t(In_t[j*2]))

    feature_t_ = Concatenate(axis=1)(feature_t)

    relation1_t = []
    for j in range(n_data):
        T = Lambda(lambda x: K.batch_dot(x[0],x[1]))([In_t[j*2+1],feature_t_])
        relation1_t.append(m1_t([T,T,T,In_t[n_data*2]]))

    relation1_t_ = Concatenate(axis=1)(relation1_t)

    relation2_t = []
    for j in range(n_data):
        T = Lambda(lambda x: K.batch_dot(x[0],x[1]))([In_t[j*2+1],relation1_t_])
        relation2_t.append(m2_t([T,T,T,In_t[n_data*2]]))

    V_t = []
    for j in range(n_data):
        V_t.append(q_net_t([feature_t[j],relation1_t[j],relation2_t[j]]))

    model_t = Model(input=In_t,output=V_t)
    buff=ReplayBuffer(capacity)
    times = [0]*n_data
    #append all data here, unload for each model
    models_data.append(
        [model_t, V_t, relation1_t, relation2_t, feature_t, In_t, q_net_t, m1_t, m2_t, encoder_t, 
        model, V, relation1, relation2, feature, In, q_net, m1, m2, encoder, times, data ])


#helper fcns that look awful
def load_data():
    model_t, V_t, relation1_t, relation2_t, feature_t, In_t, q_net_t, m1_t, m2_t, encoder_t, model, V, relation1, relation2, feature, In, q_net, m1, m2, encoder, times, data = models_data[curr_model]

def store_data():
    models_data[curr_model] = [
        model_t, V_t, relation1_t, relation2_t, feature_t, In_t, q_net_t, m1_t, m2_t, encoder_t, 
        model, V, relation1, relation2, feature, In, q_net, m1, m2, encoder, times, data ]

def set_model(m_to, m_from, mutate=False, p=0):
    #get data from m_from
    curr_model = m_from
    load_data()
    model_stuff = [encoder, q_net, m1, m2, encoder_t, q_net_t, m1_t, m2_t]
    saved_weights_ = [x.get_weights() for x in model_stuff]
    
    #set data in m_to
    curr_model = m_to
    load_data()
    model_stuff = [encoder, q_net, m1, m2, encoder_t, q_net_t, m1_t, m2_t]
    if not mutate:
        rand_n = np.random.uniform(0,1)
        if rand_n < p:
            print("Setting model ", m_from, " to model ", m_to)
            for i in range(len(model_stuff)):
                model_stuff[i].set_weights(saved_weights_[i])
    else:
        for i in range(len(model_stuff)):
            rand_n = np.random.uniform(0,1)
            if rand_n < p:
                print("Setting part of model ", m_from, " to model ", m_to)
                model_stuff[i].set_weights(saved_weights_[i])



rewards_evo = [0]*N_evo
load_data() #load 0-th data
loss_history_all = []
for i in range(N_evo):
    loss_history_all.append([])
#########playing#########
while(1):

    i_episode+=1	
    for i in range(n_data):
        times[i] = times[i] + 1
        if data[i].now == data[i].target:
            num+=1
            data[i].now = np.random.randint(n_router)
            data[i].target = np.random.randint(n_router)
            data[i].time = 0
            if data[i].edge != -1:
                edges[data[i].edge].load -= data[i].size
            data[i].size = np.random.rand()
            data[i].edge = -1
            total_time+=times[i]
            times[i] = 0

    obs = observation(router, edges, data, n_router, n_data, t_edge)
    adj = Adjacency(data,n_data)
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

    data, edges, reward, done = set_action(act,edges, data, n_data, t_edge)
    next_obs = observation(router, edges, data, n_router, n_data, t_edge)

    buff.add(obs, act, next_obs, reward, done, adj)

    score += sum(reward)
    print_every = K_evo/5
    if i_episode %  print_every:
        print(int(i_episode/print_every))
        print("score", score/print_every,end='\t')
        f.write(str(score/print_every)+'\t')
        if num !=0:
            print(total_time/num,end='\t')
            f.write(str(total_time/num)+'\t')
        else :
            print(0,end='\t')
            f.write(str(0)+'\t')
        print(num,end='\t')
        print(loss/print_every)
        f.write(str(num)+'\t'+str(loss/print_every)+'\n')
        rewards_evo[curr_model] += -loss #+score??
        loss_history_all[curr_model].append(loss)
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
        #rewards_evo[curr_model] = -loss
        # check if we're done; then do genetic algo
        if (curr_model+1) >= N_evo:
            print("---- LOSS LAST 100 FAR ----")
            for i in range(N_evo):
                avg_loss = np.sum(np.array(loss_history_all[i][-100:]))/100
                print("Model ", str(i), ". AVG: ", str(avg_loss), "...", str(loss_history_all[i][-5:]))
            print(" ---- ")
            # #genetic algo here
            # argmax_ind = rewards_evo.index(max(rewards_evo))
            # max_r = rewards_evo[argmax_ind]
            # for i in range(len(rewards_evo)):
            # 	r = rewards_evo[i]
            # 	probi = 1 - np.exp(eta_evo*r) / np.exp(eta_evo*max_r)
            # 	print("probi: heyy", probi)
            # 	#with prob. pi, swap model i with model argmax_ind
            # 	if np.random.random() < probi:
            # 		set_model(i, argmax_ind)

            # rewards_evo = [0]*N_evo #zero out prev rewards
            # curr_model = 0

            #genetic algo here 
            print("Genetic algorithm output: ")
            argmax_ind = rewards_evo.index(max(rewards_evo))
            max_r = rewards_evo[argmax_ind]
            diff = np.max(rewards_evo) - np.min(rewards_evo)
            l = 2 
            rewards_norm = rewards_evo/diff+0.5
            reward_exp = np.exp(rewards_norm*l)
            max_reward = reward_exp[argmax_ind]
            for i in range(len(rewards_evo)):               
                #with prob. p, swap model i with model argmax_ind. If mutate, swap parts
                print("p = ", 1-(reward_exp[i]/max_reward))
                if i != argmax_ind:
                    set_model(i, argmax_ind, mutate=False, p=1-(reward_exp[i]/max_reward))

            rewards_evo = [0]*N_evo #zero out prev rewards
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

    
