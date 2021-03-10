import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical

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

class Environment(object):
    def __init__(self, n_router, t_edge, n_data):
        self.n_router = n_router
        self.t_edge = t_edge
        self.n_data = n_data
        self.router = []
        self.edges = []
        self.data = []
        for i in range(self.n_router):
            # Add n_router routers to the graph at random locations
	        self.router.append(Router(np.random.random(),np.random.random()))
        for i in range(n_router):
            dis = []
            for j in range(n_router):
                dis.append([(self.router[j].x - self.router[i].x)**2 + (self.router[j].y - self.router[i].y)**2, j])
            dis.sort(key = lambda x: x[0],reverse = False)

            for j in range(self.n_router):
                if len(self.router[i].neighbor) == 3:
                    break
                if j == 0 :
                    continue
                if len(self.router[dis[j][1]].neighbor) < 3:
                    self.router[i].neighbor.append(dis[j][1])
                    self.router[dis[j][1]].neighbor.append(i)
                    if i<dis[j][1]:
                        self.edges.append(Edge(i,dis[j][1],np.sqrt(dis[j][0])))
                        self.router[i].edge.append(t_edge)
                        self.router[dis[j][1]].edge.append(t_edge)
                        self.t_edge += 1
                    else:
                        self.edges.append(Edge(dis[j][1],i,np.sqrt(dis[j][0])))
                        self.router[dis[j][1]].edge.append(t_edge)
                        self.router[i].edge.append(t_edge)
                        self.t_edge += 1
        for i in range(self.n_data):
	        self.data.append(Data(np.random.randint(self.n_router),np.random.randint(self.n_router),np.random.random(),i))
    def plot(self):
        for i in range(self.n_router):
            plt.scatter(self.router[i].x, self.router[i].y, color = 'orange')
        for e in self.edges:
            plt.plot([self.router[e.start].x, self.router[e.end].x], [self.router[e.start].y, self.router[e.end].y], color='black')

    def observation(self):
        obs = []
        for i in range(self.n_data):
            ob=[]
            ####meta information####
            ob.append(self.data[i].now)
            ob.append(self.data[i].target)
            ob.append(self.data[i].edge)
            ob.append(self.data[i].size)
            ob.append(self.data[i].priority)

            ####edge information####
            for j in self.router[self.data[i].now].edge:
                ob.append(j)
                ob.append(self.edges[j].start)
                ob.append(self.edges[j].end)
                ob.append(self.edges[j].len)
                ob.append(self.edges[j].load)

            ####other datas####
            count = 0
            self.data[i].neigh = []
            self.data[i].neigh.append(i)

            for j in range(self.n_data):
                if j==i:
                    continue
                if (self.data[j].now in self.router[self.data[i].now].neighbor)|(self.data[j].now == self.data[i].now):
                    count+=1
                    ob.append(self.data[j].now)
                    ob.append(self.data[j].target)
                    ob.append(self.data[j].edge)
                    ob.append(self.data[j].size)
                    ob.append(self.data[i].priority)
                    self.data[i].neigh.append(j)

                if count==3:
                    break
            for j in range(3-count):
                self.data[i].neigh.append(-1)
                for k in range(5):
                    ob.append(-1) #invalid placeholder

            obs.append(np.array(ob))

        return obs

    def set_action(self, act):
        reward = [0]*self.n_data
        done = [False]*self.n_data

        for i in range(self.n_data):
            if self.data[i].edge != -1:
                self.data[i].time -= 1
                if self.data[i].time == 0:
                    self.edges[self.data[i].edge].load -= self.data[i].size
                    self.data[i].edge = -1

            elif act[i]==0:
                continue

            else:
                t = self.router[self.data[i].now].edge[act[i]-1]
                if self.edges[t].load + self.data[i].size >1:
                    reward[i] = -0.2
                else:
                    self.data[i].edge = t
                    self.data[i].time = self.edges[t].len
                    self.edges[t].load += self.data[i].size

                    if self.edges[t].start == self.data[i].now:
                        self.data[i].now = self.edges[t].end
                    else:
                        self.data[i].now = self.edges[t].start

            if self.data[i].now == self.data[i].target:
                reward[i] = 10
                done[i] = True

        return reward, done

    def adjacency(self):
        adj = []
        for j in range(self.n_data):
            l = to_categorical(self.data[j].neigh,num_classes=self.n_data)
            for i in range(4):
                if self.data[j].neigh[i] == -1:
                    l[i]=np.zeros(self.n_data)
            adj.append(l)
        return adj