import sys

from PygameDDPG import ENVIROMENT
import gym
import pygame
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from keras import optimizers
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input, Dropout

#Class Experience_Replay_Buffer, here all the return data of the enviroment will be stored
  
class Experience_Replay_Buffer:
  def __init__(self, stateshape, max_Experience, actionspace, batch_size):
    self.max_Experience = int(max_Experience)
    self.batch_size = batch_size
    
    self.s = np.empty([self.max_Experience, stateshape])
    self.a = np.empty([self.max_Experience, actionspace])
    self.r = np.empty([self.max_Experience, 2])
    self.d = np.empty([self.max_Experience])
    
    self.pointer = 0
    self.max_Value = 0
    
  def add_Experience(self, s, a, r, d):
    self.s[self.pointer] = s
    self.a[self.pointer] = a
    self.r[self.pointer] = r
    self.d[self.pointer] = d
    
    self.pointer = (self.pointer + 1) % self.max_Experience
    self.max_Value = max(self.pointer, self.max_Value)
    
  def get_batch(self):
    indices = np.random.randint(self.max_Value - 1, size=self.batch_size)
    
    s = self.s[indices]
    a = self.a[indices]
    r = self.r[indices]
    s2 = self.s[indices + 1]
    d = self.d[indices]
    
    return s, a, r, s2, d
    
  def get_tensor(self, sarsd):
    s = tf.convert_to_tensor(sarsd[0], dtype=tf.float32)
    a = tf.convert_to_tensor(sarsd[1], dtype=tf.float32)
    r = tf.convert_to_tensor(sarsd[2], dtype=tf.float32)
    s2 = tf.convert_to_tensor(sarsd[3], dtype=tf.float32)
    d = tf.convert_to_tensor(sarsd[4], dtype=tf.float32)
    return s, a, r, s2, d





#Class for all the neural nets, it contains 4 models:
#a model for estimating a at runtime
#two models which store Q values, the lowest of the two will be used to calculate target
#one model which models Q and mu and can be used to mu
    
class Deep_Deterministic_Policy_Gradients:
  def __init__(self, name, input_shape, output_shape, hidden_layers, q_lr, mu_lr, save_filepath, tau, policy_strength, gamma):
    self.input_shape = input_shape
    self.output_shape = output_shape
    self.hidden_layers = hidden_layers
    
    self.name = name
    
    self.save_filepath=save_filepath
    self.tau = tau
    self.policy_strength = policy_strength
    self.gamma = gamma
    
    self.target_layers = []
    self.mu_layers = []
    self.q_layers = []
    for i in range(len(hidden_layers) + 1):
      self.target_layers.append(i + 1)
      self.mu_layers.append(i + 1)
    for i in range(len(hidden_layers) + 1):
      self.target_layers.append(i + 4 + len(self.hidden_layers))
      self.q_layers.append(i + 4 + len(self.hidden_layers))
    
    s = Input(shape=(input_shape))
    a = Input(shape=(output_shape))
    s2 = Input(shape=(input_shape))
    r = Input(shape=(1))
    d = Input(shape=(1))
    
    mu, q, q2, q_mu, q_mu2 = self.Q_MU(s, a, "Main_Network")
    _, _, _, target_q_mu, target_q_mu2 = self.Q_MU(s2, a, "Target_Network")
    
    self.mu_optimizer = tf.keras.optimizers.Adam(learning_rate=mu_lr)
    self.q_optimizer = tf.keras.optimizers.Adam(learning_rate=q_lr)
    
    
    target_1 = r + self.gamma * target_q_mu * (1 - d)
    target_2 = r + self.gamma * target_q_mu2 * (1 - d)
    target = tf.maximum(target_1, target_2)
    
    
    self.model1 = Model([s], mu)
    self.model2 = Model([s], q_mu)
    self.model3 = Model([s], q_mu2)
    self.model4 = Model([s, a], q)
    self.model5 = Model([s, a], q2)
    self.model6 = Model([s2], target_q_mu)
    self.model7 = Model([s2], target_q_mu2)
    self.model8 = Model([s2, r, d], target)
    
    self.mu_loss = lambda: -tf.reduce_mean(self.model2(self.mu_input))
    self.get_mu_vars = self.model1.trainable_weights
    
    self.q1_loss = lambda: tf.reduce_mean((self.model4(self.q1_input1) - self.model8(self.q1_input2))**2)
    self.get_q1_vars = self.model4.trainable_weights
    
    self.q2_loss = lambda: tf.reduce_mean((self.model5(self.q2_input1) - self.model8(self.q2_input2))**2)
    self.get_q2_vars = self.model5.trainable_weights
    
    for i in self.target_layers:
      self.model6.layers[i].set_weights(self.model2.layers[i].get_weights())
      self.model7.layers[i].set_weights(self.model3.layers[i].get_weights())
  
  #creates all the different model sorts
  def Q_MU(self, s, a, layer_name):
    mu = self.policy_strength * self.ANN(s, self.output_shape, "relu", "tanh", layer_name, "mu")
    
    q = tf.squeeze(self.ANN(tf.concat([s, a], axis=-1), 1, "relu", None, layer_name, "q_mu"))
    q2 = tf.squeeze(self.ANN(tf.concat([s, a], axis=-1), 1, "relu", None, layer_name, "q_mu2"))
    q_mu = tf.squeeze(self.ANN(tf.concat([s, mu], axis=-1), 1, "relu", None, layer_name, "q_mu"))
    q_mu2 = tf.squeeze(self.ANN(tf.concat([s, mu], axis=-1), 1, "relu", None, layer_name, "q_mu2"))
    
    return mu, q, q2, q_mu, q_mu2
    
  #used at the initialization of the class to create the nn
  def ANN(self, x, output_size, hidden_activation, output_activation, layer_name, q_mu):
    for h in self.hidden_layers:
      x = Dense(h, activation=hidden_activation, name="%s_%s_%s" % (q_mu, layer_name, h))(x)
      
    return Dense(output_size, activation=output_activation)(x)
    
  def mu_target_op(self, s):
    self.mu_input = s
    
    with tf.GradientTape() as tape:
      loss = self.mu_loss()
    vars = self.get_mu_vars
    grads = tape.gradient(loss, vars)
    grads_and_vars = zip(grads, vars)
    self.mu_optimizer.apply_gradients(grads_and_vars)
    
    return loss
    
  def q1_target_op(self, input1, input2):
    self.q1_input1 = input1
    self.q1_input2 = input2
    
    with tf.GradientTape() as tape:
      loss = self.q1_loss()
    vars = self.get_q1_vars
    grads = tape.gradient(loss, vars)
    grads_and_vars = zip(grads, vars)
    self.q_optimizer.apply_gradients(grads_and_vars)
    
    return loss
    
  def q2_target_op(self, input1, input2):
    self.q2_input1 = input1
    self.q2_input2 = input2
    
    with tf.GradientTape() as tape:
      loss = self.q2_loss()
    vars = self.get_q2_vars
    grads = tape.gradient(loss, vars)
    grads_and_vars = zip(grads, vars)
    self.q_optimizer.apply_gradients(grads_and_vars)
    
    return loss
    
  def update_weights(self):
    for i in self.mu_layers:
      self.model2.layers[i].set_weights(self.model1.layers[i].get_weights())
      self.model3.layers[i].set_weights(self.model1.layers[i].get_weights())
    
    for i in self.q_layers:
      self.model2.layers[i].set_weights(self.model4.layers[i-(len(self.hidden_layers) + 1)].get_weights())
      self.model3.layers[i].set_weights(self.model5.layers[i-(len(self.hidden_layers) + 1)].get_weights())
    
    
    
  #gets the weights from the main network and puts them in the target network
  def set_target_network(self):
    for i in self.target_layers:
      self.model6.layers[i].set_weights(self.target_network_weigths(self.model2.layers[i].get_weights(), self.tau, self.model6.layers[i].get_weights(), (1 - self.tau)))
      self.model7.layers[i].set_weights(self.target_network_weigths(self.model3.layers[i].get_weights(), self.tau, self.model7.layers[i].get_weights(), (1 - self.tau)))
      
  def target_network_weigths(self, x, y, x2, y2):
    x[0] = x[0] * y
    x2[0] = x2[0] * y2
    new_x = [x[0] + x2[0], x[1]]
    return new_x
    
  def save_all_weights(self, filepath):
    self.model1.save_weights(filepath=self.save_filepath + "%s" % (0))
    self.model2.save_weights(filepath=self.save_filepath + "%s" % (1))
    self.model3.save_weights(filepath=self.save_filepath + "%s" % (2))
    self.model4.save_weights(filepath=self.save_filepath + "%s" % (3))
    self.model5.save_weights(filepath=self.save_filepath + "%s" % (4))
    self.model6.save_weights(filepath=self.save_filepath + "%s" % (5))
    self.model7.save_weights(filepath=self.save_filepath + "%s" % (6))
    self.model8.save_weights(filepath=self.save_filepath + "%s" % (7))
  
  def load_all_weights(self, filepath):
    self.model1.load_weights(filepath=filepath + "%s" % (0))
    self.model2.load_weights(filepath=filepath + "%s" % (1))
    self.model3.load_weights(filepath=filepath + "%s" % (2))
    self.model4.load_weights(filepath=filepath + "%s" % (3))
    self.model5.load_weights(filepath=filepath + "%s" % (4))
    self.model6.load_weights(filepath=filepath + "%s" % (5))
    self.model7.load_weights(filepath=filepath + "%s" % (6))
    self.model8.load_weights(filepath=filepath + "%s" % (7))
    
  def save_policy_model(self, filepath, i):
    self.model1.save("policy_model\policy_model%s_%s" % (i, self.name))



  
  
  
#returns an action

def get_action(s, noise, network):
  return np.clip(network.model1.predict(s.reshape(1, stateshape))[0] + np.clip(noise, -noise_clip, noise_clip), -action_clip, action_clip)
  
  
  
#plays the game to gain experience

def play_game(act_fn):
  done = False
  s = env.reset(env_iters)
  rewards = 0
  connections = 0
  while not done:
    noise = np.random.randn(2) * noise_scale
    a = act_fn(s, noise, ddpg1)
    a2 = act_fn(s, noise, ddpg2)
    both_a = [a, a2]
    s2, r, done = env.move(both_a)
    a = np.concatenate((a, a2), axis=0)
    s2 = s2
    exp_buffer.add_Experience(s, a, r, done)
    
    rewards += r[0]
    connections += r[0]>0
    s = s2

  return rewards, connections
  
  
  
#plays the game with display enabled
  
def test(act_fn):
  env.set_display(True)
  done = False
  rewards = 0
  
  s = env.reset(300)
  while not done:
    a = act_fn(s, 0, ddpg1)
    a2 = act_fn(s, 0, ddpg2)
    a = [a, a2]
    s2, r, d = env.move(a)
    s2 = s2
    
    rewards += r[0]
    if d:
      done = True
      
    s = s2
      
    time.sleep(0.01)
  env.set_display(False)
  print("test rewards:%.3f" % (rewards))



#the main loop

def main_loop(
    epochs,
    replay_size,
    env_iters,
    gamma,
    start_steps,
    policy_delay,
    save_freq,
    load_file,
    noise_scale,
    noise_reduction,
    do_tests_every,
    train_steps_per_epoch,
    act_fn):

  #first check if an old model should be loaded
  if load_file:
    load_filepath = r"C:\Users\bjarn\Programming\Reinforcement_learning\Soccer\Saved_Weights\TD3weights"
    ddpg.load_all_weights(load_filepath)
    
  #initialize lists to keep track of rewards
  total_rewards = np.empty([epochs])
  total_avg_reward = np.empty([epochs])
  total_connections = np.empty([epochs])
  
  #init other variables
  t0 = datetime.now()

  for i in range(epochs):
    t2 = datetime.now()
    rewards, connections = play_game(act_fn)
    noise_scale -= noise_reduction
    
    total_connections[i] = connections
    total_rewards[i] = rewards
    total_avg_reward[i] = np.mean(total_rewards[max(0, i-10):i+1])
    
    avg_loss = []
    
    if do_tests_every is not None:
      if i % do_tests_every == 0 and i != 0:
        test(act_fn)
      
    if i % save_freq == 0 and i != 0:
      ddpg1.save_policy_model(save_filepath, i)
      ddpg2.save_policy_model(save_filepath, i)
    
    if i == start_steps:
      act_fn = None
      act_fn = get_action
      print ("using own policy now")
    
    
    for iter in range(train_steps_per_epoch):
      s, a, r, s2, d = exp_buffer.get_tensor(exp_buffer.get_batch())
      input1 = [s, tf.transpose(tf.transpose(a)[:2])]
      input2 = [s2, tf.transpose(tf.transpose(r)[0]), d]
      
      input1_2 = [s, tf.transpose(tf.transpose(a)[2:])]
      input2_2 = [s2, tf.transpose(tf.transpose(r)[1]), d]
      
      q_loss = ddpg1.q1_target_op(input1, input2)
      ddpg1.q2_target_op(input1, input2)
      
      q_loss = ddpg2.q1_target_op(input1_2, input2_2)
      ddpg2.q2_target_op(input1_2, input2_2)
      
      ddpg1.update_weights()
      ddpg2.update_weights()
      
      if iter % policy_delay == 0:
        ddpg1.mu_target_op(s)
        ddpg1.set_target_network()
        
        ddpg2.mu_target_op(s)
        ddpg2.set_target_network()
        
      avg_loss.append(q_loss)
    
    t3 = datetime.now()
    print ("iteration:%s, rewards:%.2f, noise_scale:%.2f, avg_reward_last_10:%.3f, time:%s, loss:%.3f" % (i, rewards, noise_scale, total_avg_reward[i], t3 - t2, np.mean(avg_loss)), "\n")
      
    
  #ddpg1.save_all_weights(save_filepath)
  
  t1 = datetime.now()
  
  print ("total time training %s" % (t1 - t0))
  plt.plot(total_rewards, label="Total Rewards")
  plt.plot(total_avg_reward, label="Total Average Rewards")
  plt.legend()
  plt.show()
  
  
  
if __name__ == "__main__":
  #define all the main hyperparameters
  epochs=1500
  train_steps_per_epoch=200
  env_iters=3600
  replay_size=int(1e6)
  gamma=0.99
  tau = 0.005
  mu_lr=5e-4
  q_lr=5e-4
  action_clip=2
  noise_clip=0.5
  batch_size=128
  start_steps=40
  policy_delay=2
  policy_strength=2
  save_freq=50
  load_file=False
  do_tests_every=5
  hidden_layers=[50, 250, 100]
  noise_scale = 0.1
  noise_reduction = (noise_scale-1e-4) / epochs
  save_filepath=r"C:\Users\bjarn\Programming\Reinforcement_learning\Soccer\Saved_Weights\TD3weights"
  actionspace = 2
  act_fn = lambda s, noise, network : np.clip(np.random.randn(actionspace) * policy_strength, -action_clip, action_clip)
  
  
  #Initialize the main classes
  #env = ENVIROMENT(False)
  x, y, x2, y2, b_x, b_y = 550, 250, 250, 250, 400, 250
  player_1 = {"x": x, "y": y, "color": (0, 0, 250, 1), "mode": "bot"}
  player_2 = {"x": x2, "y": y2, "color": (250, 0, 0, 1), "mode": "bot"}
  ball = {"x": b_x, "y": b_y}
  env = ENVIROMENT(False, env_iters, (player_1, player_2), ball, True)
  
  #create the pygame enviroment
  pygame.init()
  
  #init shapes for the buffer and the buffer
  
  stateshape = len(env.reset(env_iters))
  exp_buffer = Experience_Replay_Buffer(stateshape, replay_size, actionspace * 2, batch_size)
  
  #create the ddpg models
  name1 = "model1"
  name2 = "model2"
  
  ddpg1 = Deep_Deterministic_Policy_Gradients(name1, stateshape, actionspace, hidden_layers, q_lr, mu_lr, save_filepath, tau, policy_strength, gamma)
  ddpg2 = Deep_Deterministic_Policy_Gradients(name2, stateshape, actionspace, hidden_layers, q_lr, mu_lr, save_filepath, tau, policy_strength, gamma)
  
  
  main_loop(
    epochs=epochs,
    replay_size=replay_size,
    env_iters=env_iters,
    gamma=gamma,
    start_steps=start_steps,
    policy_delay=policy_delay,
    save_freq=save_freq,
    noise_scale=noise_scale,
    noise_reduction=noise_reduction,
    load_file=load_file,
    do_tests_every=do_tests_every,
    train_steps_per_epoch=train_steps_per_epoch,
    act_fn=act_fn
    )
