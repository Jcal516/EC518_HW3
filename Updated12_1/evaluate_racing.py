#Initialize carla in this code and complete the main function
from deepq import evaluate
import train_racing
import glob
import os
import sys
from reward import reward_function
from deepq import learn

try:
    egg_path = glob.glob('../../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0]
    sys.path.append(egg_path)
    print(f"Adding Carla path: {egg_path}")
except IndexError:
    print("Carla egg file not found.")
    sys.exit(1)  # Exit if Carla is not found

import carla

def main():
    """ 
    Evaluate a trained Deep Q-Learning agent 
    """ 
    env = train_racing.CarlaEnv()
    evaluate(env=env)


if __name__ == '__main__':
    main()
