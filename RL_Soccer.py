import numpy as np
import pygame
import tensorflow as tf

from PygameDDPG import mainloop, ENVIROMENT

#extra classes for convenience
class Rect:
  def __init__(self, x, y, width, height, color, action=None):
    self.x = x
    self.y = y
    self.width = width
    self.height = height
    
    self.color = color
    
    self.info = (self.x - self.width / 2, self.y - self.height / 2, self.width, self.height)
    
    self.action = action
    self.image = None





#
def check_pres(mouse):
  pos = mouse
  for b in runner.active_buttons:
    if pos[0] > b.x - b.width / 2 and pos[0] < b.x + b.width / 2 and pos[1] > b.y - b.height / 2 and pos[1] < b.y + b.height / 2:
      b.action()

def key_input(key, main_game):
  a = np.array([0, 0])
  if key[pygame.K_w]:
    a += np.array([0, -2])
  if key[pygame.K_a]:
    a += np.array([-2, 0])
  if key[pygame.K_s]:
    a += np.array([0, 2])
  if key[pygame.K_d]:
    a += np.array([2, 0])
  if key[pygame.K_ESCAPE]:
    if main_game.pause_delay == 0:
      main_game.is_paused = not main_game.is_paused
      main_game.pause_delay = 10
    
  main_game.pause_delay = np.clip(main_game.pause_delay-1, 0, 10)
    
  return a



#the main classes for running the game
class Main_menu:
  def __init__(self, win, runner, main_game):
    self.win = win
    self.runner = runner
    self.main_game = main_game
    
    start = lambda: runner.change_screen("game")
    replay = lambda: runner.change_screen("replay")
    info = lambda: runner.change_screen("info")
    quit = lambda: runner.quit()
    change_input = lambda: runner.change_input()
    increase = lambda: self.difficulty(1)
    decrease = lambda: self.difficulty(-1)
    
    self.start_button = Rect(x=400, y=200, width=300, height=100, color=(250, 0, 0, 1), action=start)
    self.replay_button = Rect(x=400, y=300, width=225, height=75, color=(0, 250, 0, 1), action=replay)
    self.info_button = Rect(x=400, y=370, width=225, height=75, color=(0, 250, 0, 1), action=info)
    self.quit_button = Rect(x=400, y=440, width=225, height=65, color=(255, 0, 0, 1), action=quit)
    self.input_mode_button = Rect(x=595, y=225, width=40, height=40, color=(255, 0, 0, 1), action=change_input)
    self.increase_dif = Rect(x=630, y=175, width=15, height=40, color=(250, 0, 0, 1), action=increase)
    self.decrease_dif = Rect(x=560, y=175, width=15, height=40, color=(250, 0, 0, 1), action=decrease)
    
    self.start_button.image = pygame.transform.smoothscale(self.runner.start, (self.start_button.width, self.start_button.height))
    self.replay_button.image = pygame.transform.smoothscale(self.runner.replay_img, (self.replay_button.width, self.replay_button.height))
    self.info_button.image = pygame.transform.smoothscale(self.runner.info_img, (self.info_button.width, self.info_button.height))
    self.quit_button.image = pygame.transform.smoothscale(self.runner.quit_img, (self.quit_button.width, self.quit_button.height))
    self.input_mode_button.image = pygame.transform.smoothscale(self.runner.mouse_img, (self.input_mode_button.width, self.input_mode_button.height))
    
    self.increase_dif.image = pygame.transform.flip(self.runner.arrow, True, False)
    self.decrease_dif.image = self.runner.arrow
    
    self.difficulty_display = Rect(x=595, y=175, width=40, height=40, color=(250, 0, 0, 1))
    self.difficulty_display.image = pygame.transform.smoothscale(self.runner.border_img, (self.difficulty_display.width, self.difficulty_display.height))
    self.difficulty_text = self.main_game.font.render(str(main_game.difficulty), False, (200, 200, 200, 1))
    
    self.buttons = [self.start_button, self.replay_button, self.info_button, self.quit_button, self.input_mode_button, self.increase_dif, self.decrease_dif]
    
  def draw(self, mouse, key, _):
    self.input(mouse, key)
    for b in self.buttons:
      if b.image is None:
        pygame.draw.rect(self.win, b.color, b.info)
      else:
        self.win.blit(b.image, (b.x - b.width/2, b.y - b.height/2))
    self.win.blit(self.difficulty_display.image, (self.difficulty_display.x - self.difficulty_display.width/2, self.difficulty_display.y - self.difficulty_display.height/2))
    self.win.blit(self.difficulty_text, (self.difficulty_display.x - self.main_game.text_offset, self.difficulty_display.y - 17))
    
  def input(self, mouse, key):
    if mouse is not None:
      check_pres(mouse)
      
  def difficulty(self, change):
    self.main_game.change_difficulty(change, False)
    self.difficulty_text = self.main_game.font.render(str(self.main_game.difficulty), False, (200, 200, 200, 1))
      
    
class Main_game:
  def __init__(self, win, runner):
    self.win = win
    self.runner = runner
    
    self.env = self.create_env()
    
    self.difficulty = 0
    
    self.current_file = self.difficulty * 100 + 50
    filepath = "policy_model\policy_model%s_model2" % (self.current_file)
    self.model = tf.keras.models.load_model(filepath)
    
    restart = lambda: self.env.reset(self.runner.game_len)
    back = lambda: runner.change_screen("menu")
    pause = lambda: self.pause()
    increase = lambda: self.change_difficulty(1)
    decrease = lambda: self.change_difficulty(-1)
    change_input = lambda: runner.change_input()
    
    
    self.back_button = Rect(x=770, y=30, width=40, height=40, color=(250, 0, 0, 1), action=back)
    self.restart_button = Rect(x=720, y=30, width=40, height=40, color=(250, 0, 0, 1), action=restart)
    self.pause_button = Rect(x=670, y=30, width=40, height=40, color=(250, 0, 0, 1), action=pause)
    self.input_mode_button = Rect(x=750, y=470, width=40, height=40, color=(255, 0, 0, 1), action=change_input)
    self.increase_dif = Rect(x=705, y=470, width=15, height=40, color=(250, 0, 0, 1), action=increase)
    self.decrease_dif = Rect(x=635, y=470, width=15, height=40, color=(250, 0, 0, 1), action=decrease)
    
    self.back_button.image = pygame.transform.smoothscale(self.runner.back_img, (self.back_button.width, self.back_button.height))
    self.restart_button.image = pygame.transform.smoothscale(self.runner.restart_img, (self.restart_button.width, self.restart_button. height))
    self.pause_button.image = pygame.transform.smoothscale(self.runner.pause_img, (self.pause_button.width, self.pause_button.height))
    self.input_mode_button.image = pygame.transform.smoothscale(self.runner.mouse_img, (self.input_mode_button.width, self.input_mode_button.height))
    self.increase_dif.image = pygame.transform.flip(self.runner.arrow, True, False)
    self.decrease_dif.image = self.runner.arrow
    
    self.text_offset = 8
    self.difficulty_display = Rect(x=670, y=470, width=40, height=40, color=(250, 0, 0, 1))
    self.difficulty_display.image = pygame.transform.smoothscale(self.runner.border_img, (self.difficulty_display.width, self.difficulty_display.height))
    self.font = pygame.font.SysFont(pygame.font.get_fonts()[0], 32)
    self.difficulty_text = self.font.render(str(self.difficulty), False, (200, 200, 200, 1))
    
    self.buttons = [self.restart_button, self.back_button, self.pause_button, self.increase_dif, self.decrease_dif, self.input_mode_button]
    
    self.is_paused = False
    self.pause_delay = 20
    self.game_is_running = False
    self.start_delay = 0
    self.counter = 60
    self.score = self.env.get_score()
    
    self.s = self.env.reset(self.runner.game_len)
    
  def draw(self, mouse, key, mouse_pos):
    if not self.game_is_running:
      self.counter -= 1
      if self.counter < self.start_delay:
        self.game_is_running = True
    a = self.input(mouse, key, mouse_pos)
    if not self.is_paused and self.game_is_running:
      self.play_game(mouse, key, a)
    self.env.draw()
    for b in self.buttons:
      if b.image is None:
        pygame.draw.rect(self.win, b.color, b.info)
      else:
        self.win.blit(b.image, (b.x - b.width/2, b.y - b.height/2))
    self.win.blit(self.difficulty_display.image, (self.difficulty_display.x - self.difficulty_display.width/2, self.difficulty_display.y - self.difficulty_display.height/2))
    self.win.blit(self.difficulty_text, (self.difficulty_display.x - self.text_offset, self.difficulty_display.y - 17))
    
    
  def play_game(self, mouse, key, a):
    a2 = self.model.predict(self.s.reshape(1, 12))[0]
    combined_a = [a, a2]
    s2, r, d = self.env.move(combined_a)
    score = self.env.get_score()
    if self.score != score:
      self.counter = 40
      self.game_is_running = False
    if d:
      self.game_is_running = False
      self.counter = 60
      self.env.reset(self.runner.game_len, True)
      if score[1] > score[0]:
        self.change_difficulty(1)
    self.score = score
    self.s = s2
    
  def input(self, mouse, key, mouse_pos):
    a1 = key_input(key, self)
    
    a2 = mouse_pos
    if a2 is None:
      a2 = (0, 0)
    if mouse is not None:
      check_pres(mouse)
      
    if self.runner.active_input_mode == "mouse":
      action = ("mouse", a2)
    else:
      action = a1
    return action
      
  def pause(self):
    self.is_paused = not self.is_paused
    if not self.is_paused:
      pygame.time.wait(600)
    
  def change_difficulty(self, change, update=True):
    self.difficulty += change
    self.difficulty = np.clip(self.difficulty, 0, 9)
    self.text_offset = 8 + (self.difficulty == 10)*8
    self.difficulty_text = self.font.render(str(self.difficulty), False, (200, 200, 200, 1))
    
    self.current_file = self.difficulty * 100 + 50
    filepath = "policy_model\policy_model%s_model2" % (self.current_file)
    self.model = tf.keras.models.load_model(filepath)
    
    self.env.reset(self.runner.game_len)
    if update:
      self.env.draw()
    
  def create_env(self):
    x, y, x2, y2, b_x, b_y = 550, 250, 250, 250, 400, 250
    player_1 = {"x": x, "y": y, "color": (0, 0, 250, 1), "mode": "player"}
    player_2 = {"x": x2, "y": y2, "color": (250, 0, 0, 1), "mode": "bot"}
    ball = {"x": b_x, "y": b_y}
    env = ENVIROMENT(False, self.runner.game_len, (player_1, player_2), ball)
    env.set_win(self.win)
    
    return env
  
  
class Replay_game(Main_game):
  def __init__(self, win, runner):
    super().__init__(win, runner)
    
    self.buttons.remove(self.input_mode_button)
    filepath2 = "policy_model\policy_model%s_model1" % (self.current_file)
    self.model2 = tf.keras.models.load_model(filepath2)
    
  def draw(self, mouse, key, _):
    self.input(mouse, key, _)
    a = self.model2.predict(self.s.reshape(1, 12))[0]
    if not self.is_paused:
      self.play_game(mouse, key, a)
    self.env.draw()
    for b in self.buttons:
      if b.image is None:
        pygame.draw.rect(self.win, b.color, b.info)
      else:
        self.win.blit(b.image, (b.x - b.width/2, b.y - b.height/2))
    self.win.blit(self.difficulty_display.image, (self.difficulty_display.x - self.difficulty_display.width/2, self.difficulty_display.y - self.difficulty_display.height/2))
    self.win.blit(self.difficulty_text, (self.difficulty_display.x - self.text_offset, self.difficulty_display.y - 17))
    
  def create_env(self):
    x, y, x2, y2, b_x, b_y = 550, 250, 250, 250, 400, 250
    player_1 = {"x": x, "y": y, "color": (0, 0, 250, 1), "mode": "bot"}
    player_2 = {"x": x2, "y": y2, "color": (250, 0, 0, 1), "mode": "bot"}
    ball = {"x": b_x, "y": b_y}
    env = ENVIROMENT(False, self.runner.game_len, (player_1, player_2), ball)
    env.set_win(self.win)
    
    return env
  
  
class Game_info:
  def __init__(self, win, runner):
    self.win = win
    self.runner = runner
    
    back = lambda: runner.change_screen("menu")
    self.back_button = Rect(x=770, y=30, width=40, height=40, color=(250, 0, 0, 1), action=back)
    self.back_button.image = pygame.transform.smoothscale(self.runner.back_img, (self.back_button.width, self.back_button.height))
    
    self.buttons = [self.back_button]
    
    
  def draw(self, mouse, keys, _):
    self.input(mouse, keys)
    for b in self.buttons:
      if b.image is None:
        pygame.draw.rect(self.win, b.color, b.info)
      else:
        self.win.blit(b.image, (b.x - b.width/2, b.y - b.height/2))
    #self.blit_text()
      
  def input(self, mouse, keys):
    if mouse is not None:
      check_pres(mouse)





class Gamerunner:
  def __init__(self):
    self.win = pygame.display.set_mode((800, 500))
    self.game_background = pygame.image.load("Soccer_Images\Soccer_Background.png")
    self.main_background = pygame.image.load("Soccer_Images\Soccer_StartBackground.png")
    self.info_background = pygame.image.load("Soccer_Images\Soccer_InfoBackground.png")
    self.arrow = pygame.image.load("Soccer_Images\Arrow.png")
    self.start = pygame.image.load("Soccer_Images\Start_Button.png")
    self.replay_img = pygame.image.load("Soccer_Images\Replay_Training.png")
    self.info_img = pygame.image.load("Soccer_Images\Game_Info_Button.png")
    self.quit_img = pygame.image.load("Soccer_Images\Quit_Button.png")
    self.restart_img = pygame.image.load("Soccer_Images\Restart_Button.png")
    self.back_img = pygame.image.load("Soccer_Images\Back_Button.png")
    self.pause_img = pygame.image.load("Soccer_Images\Pause_Button.png")
    self.border_img = pygame.image.load("Soccer_images\Border.png")
    self.mouse_img = pygame.image.load("Soccer_images\Mouse_Button.png")
    self.WASD_img = pygame.image.load("Soccer_images\WASD_Button.png")
    
    self.info_background = pygame.transform.smoothscale(self.info_background, (800, 500))
    
    self.done = False
    self.game_speed = 10
    self.mouse_is_down = False
    
    self.game_len = 1800
    
    self.main_game = Main_game(self.win, self)
    self.replay_game = Replay_game(self.win, self)
    self.mm = Main_menu(self.win, self, self.main_game)
    self.game_info = Game_info(self.win, self)
  
    self.active_screen = self.mm
    self.active_buttons = self.active_screen.buttons
    self.active_backgrounds = self.main_background
    self.active_input_mode = "mouse"
    
    pygame.display.set_caption("Game")
    
  def mainloop(self):
    self.done = False
    mouse = None
    key = []
    while not self.done:
      self.done, mouse, key, mouse_pos = self.check_done()
      self.win.blit(self.active_backgrounds, (0, 0))
      self.active_screen.draw(mouse, key, mouse_pos)
      pygame.display.update()
      pygame.time.wait(self.game_speed)
      
    pygame.quit()
      
  def check_done(self):
    done = False
    mouse = None
    mouse_pos = None
    key = pygame.key.get_pressed()
    for event in pygame.event.get():
      if event.type == pygame.MOUSEBUTTONDOWN:
        self.mouse_is_down = True
      if event.type == pygame.MOUSEBUTTONUP:
        mouse = pygame.mouse.get_pos()
        self.mouse_is_down = False
      if event.type == pygame.QUIT:
        done = True
      else:
        done = False
    if self.mouse_is_down:
      mouse_pos = pygame.mouse.get_pos()
        
    return done, mouse, key, mouse_pos
        
  def quit(self):
    self.done = True
    
  def change_screen(self, new):
    if new == "game":
      self.active_screen = self.main_game
      self.active_backgrounds = self.game_background
    elif new == "menu":
      self.active_screen = self.mm
      self.active_screen.difficulty(0)
      self.active_backgrounds = self.main_background
    elif new == "replay":
      self.active_screen = self.replay_game
      self.active_backgrounds = self.game_background
    elif new == "info":
      self.active_screen = self.game_info
      self.active_backgrounds = self.info_background
      
    self.active_buttons = self.active_screen.buttons
    
  def change_input(self):
    if self.active_input_mode == "mouse":
      self.active_input_mode = "keyboard"
      self.main_game.input_mode_button.image = pygame.transform.smoothscale(self.WASD_img, (self.main_game.input_mode_button.width, self.main_game.input_mode_button.height))
      self.mm.input_mode_button.image = pygame.transform.smoothscale(self.WASD_img, (self.mm.input_mode_button.width, self.mm.input_mode_button.height))
    else:
      self.active_input_mode = "mouse"
      self.main_game.input_mode_button.image = pygame.transform.smoothscale(self.mouse_img, (self.main_game.input_mode_button.width, self.main_game.input_mode_button.height))
      self.mm.input_mode_button.image = pygame.transform.smoothscale(self.mouse_img, (self.mm.input_mode_button.width, self.mm.input_mode_button.height))

if __name__ == "__main__":
  #x, y, x2, y2, b_x, b_y = 550, 250, 250, 250, 400, 250
  #player_1 = {"x": x, "y": y, "color": (0, 0, 250, 1)}
  #player_2 = {"x": x2, "y": y2, "color": (250, 0, 0, 1)}
  #ball = {"x": b_x, "y": b_y}
  #env = ENVIROMENT(True, 5000, (player_1, player_2), ball)
  #current_file = 950
  #filepath1 = "policy_model\policy_model%s_model2" % (current_file)
  #model = tf.keras.models.load_model(filepath1)
  
  pygame.init()
  
  runner = Gamerunner()
  runner.mainloop()