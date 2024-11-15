# # import pyspiel

# # # print(pyspiel.registered_names())

# # game = pyspiel.load_game("chess")
# # print(game)

# # print(game.num_players())
# # print(game.max_utility())
# # print(game.min_utility())
# # print(game.num_distinct_actions())

# # state = game.new_initial_state()
# # print(state)
# # print(state.current_player())
# # print(state.is_terminal())
# # print(state.returns())
# # print(state.legal_actions())

# # state = game.new_initial_state()
# # state.apply_action(1)
# # print(state)
# # print(state.current_player())
# # state.apply_action(2)
# # state.apply_action(4)
# # state.apply_action(0)
# # state.apply_action(7)
# # print(state)
# # print(state.is_terminal())
# # print(state.player_return(0))   # win for x (player 0)
# # print(state.current_player())

# from absl import app
# from absl import flags

# from open_spiel.python.algorithms.alpha_zero import alpha_zero
# from open_spiel.python.utils import spawn

# flags.DEFINE_string("path", None, "Where to save checkpoints.")
# FLAGS = flags.FLAGS


# def main(unused_argv):
#   config = alpha_zero.Config(
#       game="chess",
#       path=FLAGS.path,
#       learning_rate=0.01,
#       weight_decay=1e-4,
#       train_batch_size=128,
#       replay_buffer_size=2**14,
#       replay_buffer_reuse=4,
#       max_steps=25,
#       checkpoint_freq=25,

#       actors=4,
#       evaluators=4,
#       uct_c=1,
#       max_simulations=20,
#       policy_alpha=0.25,
#       policy_epsilon=1,
#       temperature=1,
#       temperature_drop=4,
#       evaluation_window=50,
#       eval_levels=7,

#       nn_model="resnet",
#       nn_width=128,
#       nn_depth=2,
#       observation_shape=None,
#       output_size=None,

#       quiet=True,
#   )
#   alpha_zero.alpha_zero(config)


# if __name__ == "__main__":
#   with spawn.main_handler():
#     app.run(main)


from open_spiel.python import rl_environment
from open_spiel.python import rl_tools
from open_spiel.python.algorithms import tabular_qlearner

# Create the environment
env = rl_environment.Environment("chess")
num_players = env.num_players
num_actions = env.action_spec()["num_actions"]

# Create the agents
agents = [
    tabular_qlearner.QLearner(player_id=idx, num_actions=num_actions)
    for idx in range(num_players)
]

# Train the Q-learning agents in self-play.
for cur_episode in range(1000):
  if cur_episode % 10 == 0:
    print(f"Episodes: {cur_episode}")
  time_step = env.reset()
  while not time_step.last():
    player_id = time_step.observations["current_player"]
    agent_output = agents[player_id].step(time_step)
    time_step = env.step([agent_output.action])
  # Episode is over, step all agents with final info state.
  for agent in agents:
    agent.step(time_step)
print("Done!")

# Evaluate the Q-learning agent against a random agent.
from open_spiel.python.algorithms import random_agent
eval_agents = [agents[0], random_agent.RandomAgent(1, num_actions, "Entropy Master 2000") ]

time_step = env.reset()
while not time_step.last():
  print("")
  print(env.get_state)
  player_id = time_step.observations["current_player"]
  # Note the evaluation flag. A Q-learner will set epsilon=0 here.
  agent_output = eval_agents[player_id].step(time_step, is_evaluation=True)
  print(f"Agent {player_id} chooses {env.get_state.action_to_string(agent_output.action)}")
  time_step = env.step([agent_output.action])

print("")
print(env.get_state)
print(time_step.rewards)