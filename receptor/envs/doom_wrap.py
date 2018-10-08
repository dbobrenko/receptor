from gym import spaces
from vizdoom import DoomGame, GameVariable


def make_mpdoom_fn(cfg, name='Player1', color='0',
                 map='map01', host=True, dm=True, port=None, num_players=7):
    return lambda: MPDoom(cfg=cfg, name=name, color=color, map=map, host=host,
                          dm=dm, port=port, num_players=num_players)


def make_doom_fn(cfg, map='map01', name='Player1', color='0'):
    return lambda: Doom(cfg=cfg, map=map, name=name, color=color)


class MPDoom(object):
    def __init__(self, cfg, name='Player1', color='0',
                 host=True, map='map01', dm=True, port=None, num_players=7):
        game = DoomGame()
        game.load_config(cfg)
        game_args = ""
        if host:
            # This machine will function as a host for a multiplayer game with this many
            # players (including this machine). It will wait for other machines to connect using
            # the -join parameter and then start the game when everyone is connected.
            game_args += "-host %s " % num_players
            # The game (episode) will end after this many minutes have elapsed.
            game_args += "+timelimit 10.0 "
            # Players will respawn automatically after they die.
            game_args += "+sv_forcerespawn 1 "
            # Autoaim is disabled for all players.
            game_args += "+sv_noautoaim 1 "
            # Players will be invulnerable for two second after spawning.
            game_args += "+sv_respawnprotect 1 "
            # Players will be spawned as far as possible from any other players.
            game_args += "+sv_spawnfarthest 1 "
            # Disables crouching.
            game_args += "+sv_nocrouch 1 "
            # Sets delay between respanws (in seconds).
            game_args += "+viz_respawn_delay 10 "
            game_args += "+viz_nocheat 1"
            if dm:
                # Deathmatch rules are used for the game.
                game_args += " -deathmatch"
            if port is not None:
                game_args += " -port %s" % port
        else:
            game_args += " -join 127.0.0.1"
            if port is not None:
                game_args += ":%s" % port
        game_args += " -name %s" % name
        game_args += " -colorset %s" % color
        game.add_game_args(game_args)
        game.set_death_penalty(1)
        game.set_doom_map(map)
        self.env = game
        self.observation_space = spaces.Box(0, 255, game.get_screen_size())
        self.action_space = spaces.Discrete(game.get_available_buttons_size())
        self.reward_range = None

    def step(self, action):
        # if self.dm and self.env.is_player_dead():
        #     self.env.respawn_player()
        self.env.make_action(action)
        obs = self.env.get_state()
        if obs is not None:
            obs = obs.screen_buffer
        reward = self.env.get_last_reward()
        term = self.env.is_episode_finished()
        return obs, reward, term, {}

    def reset(self):
        self.env.new_episode()
        return self.env.get_state().screen_buffer

    def frags(self):
        return self.env.get_game_variable(GameVariable.FRAGCOUNT)


class Doom(object):
    def __init__(self, cfg, name='Player1', color='0', map='map01'):
        game = DoomGame()
        game_args = ""
        game_args += " -name %s" % name
        game_args += " -colorset %s" % color
        game.add_game_args(game_args)
        game.load_config(cfg)
        game.set_death_penalty(1)
        game.set_doom_map(map)
        self.env = game
        self.observation_space = spaces.Box(0, 255, game.get_screen_size())
        self.action_space = spaces.Discrete(game.get_available_buttons_size())
        self.reward_range = None

    def step(self, action):
        self.env.make_action(action)
        obs = self.env.get_state()
        if obs is not None:
            obs = obs.screen_buffer
        reward = self.env.get_last_reward()
        term = self.env.is_episode_finished()
        return obs, reward, term, {}

    def reset(self):
        self.env.new_episode()
        return self.env.get_state().screen_buffer
