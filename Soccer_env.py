import pygame
import numpy as np
import math
import tensorflow as tf
import sys

class Player:
  def __init__(self, win, boundary, player, noise):
    self.x = player["x"] + noise()
    self.y = player["y"] + noise()
    self.vel_x = 0
    self.vel_y = 0
    self.radius = 13
    self.color1 = player["color"]
    self.color2 = (0, 0, 0, 1)
    self.win = win
    self.boundary = boundary
    self.goal = [50, 750, 200, 300]
    self.reward = 0
    self.mode = player["mode"]
    self.speed = 0
    self.action = (0, 0)
    
  def draw(self):
    pygame.draw.circle(self.win, self.color2, (int(self.x), int(self.y)), self.radius + 3)
    pygame.draw.circle(self.win, self.color1, (int(self.x), int(self.y)), self.radius - 1)
    
  def move(self, action):
    if self.mode == "bot":
      self.reward = 0
      self.vel_x += action[0]
      self.vel_y += action[1]
      self.last_vel_x, self.last_vel_y = self.vel_x, self.vel_y
      if self.vel_x < 0.1 and self.vel_x > -0.1:
        self.vel_x = 0
      if self.vel_y < 0.1 and self.vel_y > -0.1:
        self.vel_y = 0
      self.vel_x /= 1.18
      self.vel_y /= 1.18
      #self.vel_x += ((self.vel_x > 0) * -0.1 + (self.vel_x > 0) * -0.02 * get_speed(self.vel_x, self.vel_y)) + ((self.vel_x < 0) * 0.1 + (self.vel_x < 0) * 0.02 * get_speed(self.vel_x, self.vel_y))
      #self.vel_y += ((self.vel_y > 0) * -0.1 + (self.vel_y > 0) * -0.02 * get_speed(self.vel_x, self.vel_y)) + ((self.vel_y < 0) * 0.1 + (self.vel_y < 0) * 0.02 * get_speed(self.vel_x, self.vel_y))
      self.vel_x += -self.vel_x * (self.x < self.boundary[0]) * (self.vel_x < 0) + -self.vel_x * (self.vel_x > 0) * (self.x > self.boundary[1])
      self.vel_y += -self.vel_y * (self.y < self.boundary[2]) * (self.vel_y < 0) + -self.vel_y * (self.vel_y > 0) * (self.y > self.boundary[3])
      self.x += self.vel_x
      self.y += self.vel_y
    else:
      if action[0] == "mouse":
        x_pos = action[1][0]
        y_pos = action[1][1]
        speed = 12
        vector2d = np.array([x_pos - self.x, y_pos - self.y])
        self.vel_x = vector2d[0] / np.abs(vector2d).sum() * speed
        self.vel_y = vector2d[1] / np.abs(vector2d).sum() * speed
        if (x_pos is 0 and y_pos is 0) or (np.abs(vector2d[0]) < 4 and np.abs(vector2d[1]) < 4):
          self.vel_x = 0
          self.vel_y = 0
        self.last_vel_x, self.last_vel_y = self.vel_x, self.vel_y
        
        self.x += self.vel_x
        self.y += self.vel_y
      else:
        self.reward = 0
        self.vel_x += action[0]
        self.vel_y += action[1]
        self.last_vel_x, self.last_vel_y = self.vel_x, self.vel_y
        if self.vel_x < 0.1 and self.vel_x > -0.1:
          self.vel_x = 0
        if self.vel_y < 0.1 and self.vel_y > -0.1:
          self.vel_y = 0
        self.vel_x /= 1.18
        self.vel_y /= 1.18
        #self.vel_x += ((self.vel_x > 0) * -0.1 + (self.vel_x > 0) * -0.02 * get_speed(self.vel_x, self.vel_y)) + ((self.vel_x < 0) * 0.1 + (self.vel_x < 0) * 0.02 * get_speed(self.vel_x, self.vel_y))
        #self.vel_y += ((self.vel_y > 0) * -0.1 + (self.vel_y > 0) * -0.02 * get_speed(self.vel_x, self.vel_y)) + ((self.vel_y < 0) * 0.1 + (self.vel_y < 0) * 0.02 * get_speed(self.vel_x, self.vel_y))
        self.vel_x += -self.vel_x * (self.x < self.boundary[0]) * (self.vel_x < 0) + -self.vel_x * (self.vel_x > 0) * (self.x > self.boundary[1])
        self.vel_y += -self.vel_y * (self.y < self.boundary[2]) * (self.vel_y < 0) + -self.vel_y * (self.vel_y > 0) * (self.y > self.boundary[3])
        self.x += self.vel_x
        self.y += self.vel_y
    
  def collided(self):
    self.vel_x = 0
    self.vel_y = 0
    self.speed = 0
    self.reward = 0
    
  def check_boundary(self):
    #if self.x < self.boundary[0] or self.x > self.boundary[1]:
    #  self.vel_x = -self.vel_x
    #elif self.y < self.boundary[2] or self.y > self.boundary[3]:
    #  self.vel_y = -self.vel_y
    pass
    
class Ball():
  def __init__(self, env, win, boundary, ball, noise):
    self.env = env
    self.x = ball["x"] + noise()
    self.y = ball["y"] + noise()
    self.vel_x = 0
    self.vel_y = 0
    self.radius = 10
    self.color = (200, 200, 200, 1)
    self.reduc_speed = 0.02
    self.boundary = boundary
    self.win = win
    self.give_reward = 0
    self.d = False
    self.goal = [50, 750, 200, 300]
    self.reward1 = 0
    self.reward2 = 0
    
  def draw(self):
    pygame.draw.circle(self.win, self.color, (int(self.x), int(self.y)), self.radius)
    
  def move(self):
    self.reward1 = 0
    self.reward2 = 0
    self.vel_x += -self.vel_x * (self.x < self.boundary[0]) * (self.vel_x < 0) * ((self.y > self.goal[3]) + (self.y < self.goal[2])) * 2 + -self.vel_x * (self.vel_x > 0) * (self.x > self.boundary[1]) * 2 * ((self.y > self.goal[3]) + (self.y < self.goal[2]))
    self.vel_y += -self.vel_y * (self.y < self.boundary[2]) * (self.vel_y < 0) * 2 + -self.vel_y * (self.vel_y > 0) * (self.y > self.boundary[3]) * 2
    if bool(self.x > (self.boundary[1] + 10)) + (self.x < (self.boundary[0] - 10)):
      self.env.reset(self.env.time * 60, False)
    if (self.x > (self.boundary[1] + 10)):
      self.env.set_score((1, 0))
      self.env.goal_reward1 = -100
      self.env.goal_reward2 = 100
    if (self.x < (self.boundary[0] - 10)):
      self.env.set_score((0, 1))
      self.env.goal_reward1 = 100
      self.env.goal_reward2 = -100
    self.vel_x /= 1 + self.reduc_speed
    self.vel_y /= 1 + self.reduc_speed
    self.x += self.vel_x
    self.y += self.vel_y
    
  def check_collision(self, other):
    if get_distance(self.x, self.y, self.radius, other.x, other.y, other.radius) < 0:
      speed = np.sqrt(get_speed(other.last_vel_x, other.last_vel_y))
      rot = check_rot(self.x, self.y, other.x, other.y)
      self.vel_x = -np.cos(rot) * speed
      self.vel_y = np.sin(rot) * speed
      other.collided()
      
      self.give_reward = 0
      
  def check_boundary(self):
    #if self.x < self.boundary[0] or self.x > self.boundary[1]:
    #  self.vel_x = -self.vel_x
    #elif self.y < self.boundary[2] or self.y > self.boundary[3]:
    #  self.vel_y = -self.vel_y
    pass


def get_speed(vx, vy):
  return (vx**2 + vy**2)
  
def get_distance(x, y, rad, x2, y2, rad2):
  return np.sqrt((x - x2)**2 + (y - y2)**2) - rad - rad2
  
def check_rot(x, y, x2, y2):
  dist = (x - x2) / (np.abs(x - x2) + np.abs(y - y2))
  if (y - y2) > 0:
    return np.arcsin(dist) + math.pi/2
  else:
    return np.arcsin(-dist) + math.pi/2*3
    
def check_force_rot(x, y):
  if (x + y) != 0:
    rel_x = x / (np.abs(x) + np.abs(y))
  else:
    return(0)
  #print (rel_x)
  if (y) <= 0:
    return np.arcsin(-rel_x) + math.pi/2
  else:
    return np.arcsin(rel_x) + math.pi/2*3
  
  
def class_organizer(players, b, action):
  for i, p in enumerate(players):
    p.move(action[i])
    p.check_boundary()
    b.check_collision(p)
  b.move()
  b.check_boundary()
  
def draw(players, b):
  for p in players:
    p.draw()
  b.draw()


def key_input():
  keys = pygame.key.get_pressed()
  a = np.array([0, 0])
  if keys[pygame.K_w]:
    a += np.array([0, -1])
  if keys[pygame.K_a]:
    a += np.array([-1, 0])
  if keys[pygame.K_s]:
    a += np.array([0, 1])
  if keys[pygame.K_d]:
    a += np.array([1, 0])
  if keys[pygame.K_c]:
    print(check_rot(b.x, b.y, c.x, c.y))
  if keys[pygame.K_x]:
    print(check_force_rot(c.vel_x, c.vel_y))
    
  return a
   





   
class ENVIROMENT:
  def __init__(self, display, max_iters, players, ball, create_pygame=False):
    if create_pygame:
      pygame.init()
    self.display = display
    self.background = pygame.image.load("Soccer_Images\Soccer_Background.png")
    if self.display:
      self.win = pygame.display.set_mode((800, 500))
      pygame.display.set_caption("Game")
    else:
      self.win = None
      
    self.noise = lambda: np.random.randn() * 0.1
    
    self.players = players
    self.state_normal_1 = np.tile([400, 250, 0, 0], len(self.players) + 1)
    self.state_normal_2 = np.tile([200, 100, 5, 5], len(self.players) + 1)
    
    self.ball = ball
    
    self.font = pygame.font.SysFont(pygame.font.get_fonts()[0], 32)
    self.score_text = self.font.render("0 - 0", True, (0, 0, 0, 1), (200, 200, 200, 0))
    self.left_goals = 0
    self.right_goals = 0
    
    self.reset(max_iters)
    
    self.goal_reward1 = 0
    self.goal_reward2 = 0
    
  def set_win(self, win):
    self.win = win
   
  def reset(self, max_iters, full_reset=True):
    self.boundary = [80, 720, 60, 440]
    self.player_list = []
    for i, p in enumerate(self.players):
      self.player_list.append(Player(self.win, self.boundary, p, self.noise))
    self.b = Ball(self, self.win, self.boundary, self.ball, self.noise)

    if full_reset:
      self.left_goals = 0
      self.right_goals = 0
      self.counter = 0
      self.reward = 0
      self.state = None
      self.done_flag = False 
      self.max_iters = max_iters
      
    self.set_score((0, 0))
    
    self.time = max_iters / 60
    self.set_time()
    
    return self.get_state()
    
  def move(self, action):
    self.counter += 1
    self.set_time()
    
    self.goal_reward1 = 0
    self.goal_reward2 = 0
    
    last_dist1 = get_distance(760, 250, 0, self.b.x, self.b.y, self.b.radius)
    last_dist2 = get_distance(40, 250, 0, self.b.x, self.b.y, self.b.radius)
    
    class_organizer(self.player_list, self.b, action)
  
    if self.display:
      self.draw()
      
    #for event in pygame.event.get():
    #  if event.type == pygame.QUIT:
    #    pygame.quit()
        
    if self.counter >= self.max_iters or self.b.d:
      self.done_flag = True
    else:
      self.done_flag = False
      
    self.state = self.get_state()
    self.reward1 = -get_distance(self.player_list[0].x, self.player_list[0].y, self.player_list[0].radius, self.b.x, self.b.y, self.b.radius) / 100
    self.reward2 = -get_distance(self.player_list[1].x, self.player_list[1].y, self.player_list[1].radius, self.b.x, self.b.y, self.b.radius) / 100
    
    self.reward1 += (get_distance(760, 250, 0, self.b.x, self.b.y, self.b.radius) - last_dist1) * (self.goal_reward1 == 0) + self.goal_reward1
    self.reward2 += (get_distance(40, 250, 0, self.b.x, self.b.y, self.b.radius) - last_dist2) * (self.goal_reward2 == 0) + self.goal_reward2
        
    return self.state, (self.reward1, self.reward2), self.done_flag
    
  def draw(self):
    if self.display:
      self.win.blit(self.background, (0, 0))
    draw(self.player_list, self.b)
    self.win.blit(self.score_text, (360, 10))
    self.win.blit(self.time_text, (70, 10))
    if self.display:
      pygame.display.update()
    
  def set_time(self):
    self.secs = int(self.time % 60)
    extra0 = ""
    if self.secs < 10:
      extra0 = 0
    self.time_text = self.font.render("%s:%s%s" % (int(self.time/60), extra0, self.secs), False, (200, 200, 200, 1))
    self.time -= 1 / 60
    
  def set_score(self, score):
    self.left_goals += score[0]
    self.right_goals += score[1]
    self.score_text = self.font.render(("%s - %s" % (self.left_goals, self.right_goals)), True, (0, 0, 0, 1), (200, 200, 200, 0))
    
  def get_score(self):
    return (self.left_goals, self.right_goals)
  
  def set_display(self, state):
    if state:
      self.display = True
      self.win = pygame.display.set_mode((800, 500))
      for p in self.player_list:
        p.win = self.win
      self.b.win = self.win
      pygame.display.set_caption("Game")
    else:
      self.display = False
      
  def get_state(self):
    state = [[p.x, p.y, p.vel_x, p.vel_y] for p in self.player_list]
    state = np.concatenate((state[0], state[1], [self.b.x, self.b.y, self.b.vel_x, self.b.vel_y]), axis = 0)
    state = (state - self.state_normal_1) / self.state_normal_2
    return state
    
def mainloop(env):
  done = False
  s = env.reset(2000)
  env.set_display(True)
  while not done:
    a = key_input()
    a2 = model.predict(s.reshape(1, 12))[0]
    a = (a, a2)
    s, r, d = env.move(a)
    done = env.done_flag
    
    pygame.time.wait(10)

if __name__ == "__main__":
  x = 550
  y = 250
  x2 = 250
  y2 = 250
  b_x = 400
  b_y = 250
  player_1 = {"x": x, "y": y, "color": (0, 0, 250, 1), "mode": "bot"}
  player_2 = {"x": x2, "y": y2, "color": (250, 0, 0, 1), "mode": "human"}
  ball = {"x": b_x, "y": b_y}
  env = ENVIROMENT(True, 2000, (player_1, player_2), ball, True)
  current_file = 750
  filepath1 = "policy_model\policy_model%s_model2" % (current_file)
  model = tf.keras.models.load_model(filepath1)
  mainloop(env)