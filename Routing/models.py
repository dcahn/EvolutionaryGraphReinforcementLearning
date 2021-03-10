import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape, Lambda, Flatten, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np


class MLP(tf.keras.Model):
	def __init__(self, layer_1_dim=128, layer_2_dim=128, output_dim=128):
		super(MLP, self).__init__()
		self.dense_1 = Dense(layer_1_dim, activation='relu',kernel_initializer='random_normal')
		self.dense_2 = Dense(layer_2_dim, activation='relu',kernel_initializer='random_normal')
		self.reshape = Reshape((1, output_dim))

	def call(self, inputs):
		h = self.dense_1(inputs)
		h = self.dense_2(h)
		out = self.reshape(h)
		return out

class MultiHeadsAttModel(tf.keras.Model):
	def __init__(self, l=2, d=128, dv=16, dout=128, nv=8):
		super(MultiHeadsAttModel, self).__init__()
		self.dense_1 = Dense(dv*nv, activation = "relu",kernel_initializer='random_normal')
		self.dense_2 = Dense(dv*nv, activation = "relu",kernel_initializer='random_normal')
		self.dense_3 = Dense(dv*nv, activation = "relu",kernel_initializer='random_normal')

		self.reshape_1 = Reshape((l, nv, dv))
		self.reshape_2 = Reshape((l, nv, dv))
		self.reshape_3 = Reshape((l, nv, dv))

		self.permute_1 = Lambda(lambda x: tf.keras.backend.permute_dimensions(x, (0,2,1,3)))
		self.permute_2 = Lambda(lambda x: tf.keras.backend.permute_dimensions(x, (0,2,1,3)))
		self.permute_3 = Lambda(lambda x: tf.keras.backend.permute_dimensions(x, (0,2,1,3)))

		self.batch_dot_1 = Lambda(lambda x: tf.keras.backend.batch_dot(x[0],x[1], axes=[3,3]) / np.sqrt(dv))
		self.softmax_1 = Lambda(lambda x: tf.keras.backend.softmax(x))

		self.batch_dot_2 = Lambda(lambda x: tf.keras.backend.batch_dot(x[0], x[1],axes=[3,2]))
		self.permute_4 = Lambda(lambda x: tf.keras.backend.permute_dimensions(x, (0,2,1,3)))
		self.reshape_4 = Reshape((l, dv*nv))
		self.batch_dot_3 = Lambda(lambda x: tf.keras.backend.batch_dot(x[0],x[1]))
		self.dense_4 = Dense(dout, activation = "relu",kernel_initializer='random_normal')
	
	def call(self, inputs):
		v = self.dense_1(inputs[0])
		q = self.dense_2(inputs[1])
		k = self.dense_3(inputs[2])
		ve = inputs[3]

		v1 = self.reshape_1(v)
		q1 = self.reshape_2(q)
		k1 = self.reshape_3(k)

		v2 = self.permute_1(v1)
		q2 = self.permute_2(q1)
		k2 = self.permute_3(k1)

		att = self.batch_dot_1([q2,k2])
		att1 = self.softmax_1(att)

		out = self.batch_dot_2([att1,v2])
		out1 = self.permute_4(out)
		out2 = self.reshape_4(out1)
		out3 = self.batch_dot_3([ve, out2])
		out4 = self.dense_4(out3)
		return out4

class Q_Net(tf.keras.Model):
	def __init__(self, action_dim):
		super(Q_Net, self).__init__()
		self.action_dim = action_dim
		self.flatten_0 = Flatten()
		self.flatten_1 = Flatten()
		self.flatten_2 = Flatten()
		self.concat = Concatenate()
		self.dense = Dense(action_dim,kernel_initializer='random_normal')

	def call(self, inputs):
		h0 = self.flatten_0(inputs[0])
		h1 = self.flatten_1(inputs[1])
		h2 = self.flatten_2(inputs[2])
		h =  self.concat([h0, h1, h2])
		out = self.dense(h)
		return out

def build_models(neighbors, Att_in, action_space, Q_in):
	encoder = MLP()
	m1 = MultiHeadsAttModel(l=neighbors)
	m1(Att_in)
	m2 = MultiHeadsAttModel(l=neighbors)
	m1(Att_in)
	q_net = Q_Net(action_dim = action_space)
	q_net(Q_in)
	return encoder, m1, m2, q_net

class ModelMaker():
	def __init__(self, n_data, len_feature, neighbors, Att_in, action_space, Q_in):
		self.n_data = n_data
		self.len_feature = len_feature
		self.neighbors = neighbors
		self.encoder, self.m1, self.m2, self.q_net = build_models(neighbors, Att_in, action_space, Q_in)

	def make(self):
		In= []
		for j in range(self.n_data):
			In.append(Input(shape=[self.len_feature]))
			In.append(Input(shape=(self.neighbors, self.n_data)))
		In.append(Input(shape=(1, self.neighbors)))
		feature = []
		for j in range(self.n_data):
			feature.append(self.encoder(In[j*2]))

		feature_ = Concatenate(axis=1)(feature)

		relation1 = []
		for j in range(self.n_data):
			T = Lambda(lambda x: tf.keras.backend.batch_dot(x[0],x[1]))([In[j*2+1],feature_])
			relation1.append(self.m1([T,T,T,In[self.n_data*2]]))

		relation1_ = Concatenate(axis=1)(relation1)

		relation2 = []
		for j in range(self.n_data):
			T = Lambda(lambda x: tf.keras.backend.batch_dot(x[0],x[1]))([In[j*2+1],relation1_])
			relation2.append(self.m2([T,T,T,In[self.n_data*2]]))

		V = []
		for j in range(self.n_data):
			V.append(self.q_net([feature[j],relation1[j],relation2[j]]))

		model = Model(inputs=In,outputs=V) #Model.load_model("dgn.h5")
		model.compile(optimizer=Adam(lr = 0.001), loss='mse')
		return model, V, relation1, relation2, feature, In, self.q_net, self.m1, self.m2, self.encoder