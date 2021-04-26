# AlphaGPU
Alphazero on GPU thanks to CUDA.jl

This is another implementation of alphazero algorithm, where almost everything happens on the gpu like in https://arxiv.org/abs/2104.03113
The implementation works only for connect4 and is more an exercice to learn programming CUDA in julia than a serious implementation of alphazero.

Few technical details:

-The network we train are residual dense layer, where the residual part is of the form Dense(in, in/4,relu) followed by Dense(in/4,in) followed by addition 
with entries, then relu. A typical size for in is 1024 and we use 8 such layers.

-Using a RTX 3080, it can selfplay 32000 games in parallel, with 128 rollouts (we use the regularized version of puct as in https://arxiv.org/abs/2007.12509). 
One full iteration last between 2 and 3 minutes.

-It can pit two such network for 1000 games in a few seconds (64 rollout)

-The main bottleneck beside the fact that I don't know anything about optimizing code on gpu is the card memory.


To start download the folder AlphaGPU, activate and instatiate.

Then type actor=ressimple(1024,8)|>gpu to create the network.

Then trainingPipeline(actor) to actually start training the network. All network are saved by default in the Data folder.

To load a network use JLD2 then type actor=reseau|>gpu

To test a network you can make it play by doing testvsordi(actor,rollout,player) where player=1 if you want the network to start else player=-1.

Without any tuning it matches https://github.com/jonathan-laurent/AlphaZero.jl on the tests sets in between 2 to 5 hours.

The implementation is very raw and you have to dig into the (uncommented and dirty) code if you want to tweak it.
Only simple thing you can change is:

-network size and depth

-
    
    trainingPipeline(
    
    startnet,
      
    samplesNumber = 32000, number of parallel games,
        
    rollout = 128,   rollouts
        
    iteration = 100,  number of iteration (selfplay, training, duel)

    batchsize = 2*4096,  batch size for training
    
    lr = 0.001,  learning rate for training
    
    epoch = 1,  number of epochs for training
    )


