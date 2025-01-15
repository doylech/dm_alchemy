import zmq
from dm_env import StepType
from dm_alchemy import symbolic_alchemy


class EnvironmentServer:
    def __init__(self, env_name, port=5555, verbose=False):
        self.env_name = env_name
        self.port = port
        self.verbose = verbose

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{port}")

        self.env = None

    def run(self):
        print(f"Environment server started. env_name: {self.env_name}, port: {self.port}")

        while True:
            message = self.socket.recv_json()
            command = message['command']

            # Environment does not exist
            if not self.env and command != 'new_environment':
                print("(ISSUE) No environment created yet. Please create a new environment.")
                break

            if command == 'reset':
                if self.verbose:
                    print("Resetting environment.")
                timestep = self.env.reset()
                obs = timestep.observation['symbolic_obs']
                info = None
                response = {
                    'observation': obs.tolist() if hasattr(obs, 'tolist') else obs,
                    'info': info
                }

            elif command == 'step':
                action = message['action']

                if self.verbose:
                    print(f"Step with action: {action}")

                def is_integer_like(value):
                    try:
                        # Convert to float first to handle strings with decimals like "1.0"
                        float_val = float(str(value))
                        # Check if it's equal to its integer version
                        return float_val == int(float_val)
                    except (ValueError, TypeError):
                        return False

                if is_integer_like(action):
                    action = int(action)
                else:
                    action = 0  # noop
                    print(f"Invalid action: {action}. Defaulting to 0 (noop).")

                timestep = self.env.step(action)
                obs = timestep.observation['symbolic_obs']
                reward = timestep.reward
                terminated = True if timestep.step_type == StepType.LAST else False
                truncated = False
                info = None

                response = {
                    'observation': obs.tolist() if hasattr(obs, 'tolist') else obs,
                    'reward': reward,
                    'terminated': terminated,
                    'truncated': truncated,
                    'info': info
                }

            elif command == 'close':
                if self.verbose:
                    print("Closing environment.")

                self.env.close()
                response = {'status': 'closed'}
                self.socket.send_json(response)
                break

            elif command == 'new_environment':
                level_name = message['level_name']
                seed = message['seed']

                if self.verbose:
                    print(f"Creating new environment: {level_name} with seed: {seed}")

                self.env = symbolic_alchemy.get_symbolic_alchemy_level(level_name, seed=seed, end_trial_action=True,
                                                                       num_trials=1000, max_steps_per_trial=1000)
                response = {'info': 'Environment created.'}

            else:
                raise ValueError(f"Invalid command: {command}")

            self.socket.send_json(response)


if __name__ == "__main__":
    server = EnvironmentServer(env_name="symbolic_alchemy", port=5555, verbose=True)
    server.run()
