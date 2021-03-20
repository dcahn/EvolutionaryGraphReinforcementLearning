# EvolutionaryGraphReinforcementLearning

The details of our project are listed in CS249_Final_Report.pdf, by Daniel Ahn, Tameez Latib, Mia Levy, Howard Xie. 

Our contribution is to add an evolutionary algorithm to DGN, a reinforcement learning technique with graph convolutional networks. 

The bulk of our modifications exist in routers.py, and to run the code simply type "python routers.py"
Specifically, we added tunable parameters N_evo, K_evo, and eta_evo following the evolutionary ideas. An optional parameter "mutation" in the set_model() function. Code to load and save models, and functionality of the evolutionary algorithm. The code uses a shared replay buffer since this yields best results- as seen in the paper. The bulk of the code has been modified from the original DGN paper, whose github can be found here: https://github.com/PKU-AI-Edge/DGN  

Take note of our env.yml file for creating a virtual environment. Further, our "graph_gqn_out.py" code allows us to create the graphs seen in our paper- with some adjustments needed based on the type of data.

View the paper or the presentation as files "CS249_FInal_Report.pdf" or "Presentation.pdf" respectively.