import gym
from double_dqn_cartpole import Double_dqn_cartpole
from double_dqn_grid import Double_dqn_grid
from double_dqn_mountaincar import Double_dqn_mountaincar
from dqn_cartpole import Dqn_cartpole
from dqn_grid import Dqn_grid
from dqn_mountaincar import Dqn_mountaincar
from environment import GridEnvironment

######################################################### 
# Models are saved on Pytorch Version 1.9.0 
#########################################################

IP = input('Do you want to train or test the model? (Test/Train)')
# print('train',type(train),train)
if IP.lower() == 'test':
    # DQN Grid Environment
    grid_obj = GridEnvironment('deterministic')
    dqn_grid_env = Dqn_grid(grid_obj, 'Grid Environment DQN', 25, 4)
    dqn_grid_env.neural_net.load_model('models/DQN - Grid Environment - model.pth')
    dqn_grid_env.test()
    del grid_obj
    del dqn_grid_env

    # Double DQN Grid Environment
    grid_obj = GridEnvironment('deterministic')
    main_obj_grid = Double_dqn_grid(grid_obj, 'Grid Environment DDQN', 25, 4)
    main_obj_grid.neural_net.load_model('models/Double DQN - Grid Environment - model.pth')
    main_obj_grid.test()


    # DQN Cartpole-v1
    cartpole_obj = gym.make("CartPole-v1")
    main_obj_cartpole = Dqn_cartpole(cartpole_obj, 'Cartpole Environment DQN', 4, 2)
    main_obj_cartpole.neural_net.load_model('models/DQN-CARTPOLE-model.pth')
    main_obj_cartpole.test()
    del cartpole_obj
    del main_obj_cartpole

    # DDQN Cartpole-v1
    cartpole_obj = gym.make("CartPole-v1")
    main_obj_cartpole = Double_dqn_cartpole(cartpole_obj, 'Cartpole Environment DDQN', 4, 2)
    main_obj_cartpole.neural_net.load_model('models/Double DQN - Cartpole Environment - model.pth')
    main_obj_cartpole.test()


    # DQN Mountaincar-v0
    mountaincar_obj = gym.make("MountainCar-v0")
    main_obj_mountaincar = Dqn_mountaincar(mountaincar_obj, 'Mountain Car DQN', 2, 3)
    main_obj_mountaincar.neural_net.load_model('models/DQN - Mountaincar Environment - model.pth')
    main_obj_mountaincar.test()
    del mountaincar_obj
    del main_obj_mountaincar

    # DDQN Mountaincar-v0
    mountaincar_obj = gym.make("MountainCar-v0")
    main_obj_mountaincar = Double_dqn_mountaincar(mountaincar_obj, 'Mountain Car DDQN', 2, 3)
    main_obj_mountaincar.neural_net.load_model('models/Double DQN - Mountaincar Environment - model.pth')
    main_obj_mountaincar.test()

else:
    # DQN Grid Environment
    grid_obj = GridEnvironment('deterministic')
    dqn_grid_env = Dqn_grid(grid_obj, 'Grid Environment DQN', 25, 4)
    dqn_grid_env.main()
    del grid_obj
    del dqn_grid_env

    # Double DQN Grid Environment
    grid_obj = GridEnvironment('deterministic')
    main_obj_grid = Double_dqn_grid(grid_obj, 'Grid Environment DDQN', 25, 4)
    main_obj_grid.main()


    # DQN Cartpole-v1
    cartpole_obj = gym.make("CartPole-v1")
    main_obj_cartpole = Dqn_cartpole(cartpole_obj, 'Cartpole Environment DQN', 4, 2)
    main_obj_cartpole.main()
    del cartpole_obj
    del main_obj_cartpole

    # DDQN Cartpole-v1
    cartpole_obj = gym.make("CartPole-v1")
    main_obj_cartpole = Double_dqn_cartpole(cartpole_obj, 'Cartpole Environment DDQN', 4, 2)
    main_obj_cartpole.main()


    # DQN Mountaincar-v0
    mountaincar_obj = gym.make("MountainCar-v0")
    main_obj_mountaincar = Dqn_mountaincar(mountaincar_obj, 'Mountain Car DQN', 2, 3)
    main_obj_mountaincar.main()
    del mountaincar_obj
    del main_obj_mountaincar

    # DDQN Mountaincar-v0
    mountaincar_obj = gym.make("MountainCar-v0")
    main_obj_mountaincar = Double_dqn_mountaincar(mountaincar_obj, 'Mountain Car DDQN', 2, 3)
    main_obj_mountaincar.main()