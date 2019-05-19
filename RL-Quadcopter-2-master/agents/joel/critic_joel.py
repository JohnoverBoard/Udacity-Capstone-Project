from keras import layers, models, optimizers
from keras import backend as K

class Critic():

	""" Critic (Value) Model."""
	def __init__(self, state_size, action_size):
		"""Initialize parameters and build model.

		Params
		======
			state_size (int): Dimension of each state
			action_size(int): Dimension of each action
		"""
		self.state_size = state_size
		self.action_size = action_size

		self.build_model()

	def build_model(self):
		""" Build a critic (value) network that maps (state, action) pairs ->  Q-values."""

		#Input layer
		states = layers.Input(shape=(self.state_size,), name='states')
		actions = layers.Input(shape=(self.action_size,), name='actions')

		#Hidden layers of state pathway
		net_states = layers.Dense(units=32, activation='relu')(states)
		net_states = layers.Dense(units=64, activation='relu')(net_states)

		#Hidden layers of action pathway
		net_actions = layers.Dense(units=32, activation='relu')(actions)
		net_actions = layers.Dense(units=64, activation='relu')(net_actions)

		#Combine the state pathway and action pathway
		net = layers.Add()([net_states, net_actions])
		net = layers.Activation('relu')(net)

		#Output layer
		Q_values = layers.Dense(units=1, name='q_values')(net)

		#Create Keras Model
		self.model = models.Model(input=[states, actions], outputs=Q_values)

		#Define optimizer and compile model for training
		optimizer = optimizers.Adam()
		self.model.compile(optimizer = optimizer, loss = 'mse')

		#derivative of Q values wrt actions
		action_gradients = K.gradients(Q_values, actions)

		#additional function to fetch action gradients (for actor)
		self.get_action_gradients = K.function(inputs = [self.model.input, K.learning_phase()], outputs = action_gradients)
