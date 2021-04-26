# AGZ0
This is yet another Alpha zero clone

## Basics

This project is inspired by the many replica already existing specifically:
  -https://web.stanford.edu/~surag/posts/alphazero.html
  -https://github.com/tejank10/AlphaGo.jl

It implements the core algorithm in 3 distincts files:

  -mctsworkers: a somehow multithreaded mcts(puct version) that can simulate many games in parallel

  -selfplay: handles the selfplay loop producing data and training the neural net
  
  -train: solely train the network in batch
  
 The code is as ugly as possible and mostly if not completly uncommented (self made programmer style), it will change 
 if I find the time to do so.
 
 ## Test it 
 
 You need Julia and Flux to use it.
 Clone the repository
 Choose one one files named main "name of the game here" and compile it. It will load all the necessary stuff to launch training.
 Then in the repl (or terminal? I use only Atom) type the following command
 actor=resconv2heads(64,4)       this creates the nets: 64 features, 4 residual blocks
 
 trainingPipeline(actor;nbworkers=200,game="4IARow",bufferSize=40000,samplesNumber=1000,
       rollout=400,iteration=200,cpuct=1,chkfrequency=1,batchsize=512,lr=0.0005,epoch=1,temptresh=12
       ) 
       
start training: 

-nbworkers=200 means 200 games are played in parallel

-game="4IARow" will save everything (data and net) in the folder 4IARow/Data

-bufferSize=starting size of the buffer, currently it grows by 10% every iteration to a size of 1 million position

-samplesNumber= number of game per iteration

-rollout= total number of rollout (right now I use 4 threads)

-iteration= number of training steps (here 200 mean we are generating a total of 200x1000 games)

-cpuct= puct "constant"

-chkfrequency=1 means everything is saved every iteration

-batchsize=batchsize for training 

-lr=learning rate (currently using Adam)

-epoch= training epoch in one iteration 

-temptresh=number a action before comiting to a deterministic policy(argmax)

## Results

Right now I didn't train very strongs nets.

-On gobang nets you can train a 9x9 nets on par with very good alphabeta bots in a something like 10 hours or so(to 
do so you have to start on on smaller board and use a net structure that can scale ie using pooling, this accelerate the training a lot,
a net trained on 7x7 is already very strong on 9x9)

-On 4IARow: 6 hours produce a net that can beat the perfect player when starting, though it is far from being perfect (tested on 
http://blog.gamesolver.org/solving-connect-four/02-test-protocol/ hardest set, net alone makes 5 to 10% mistake depending on the set)

-On UTTT and Reversi (6x6) it's possible to beat vanilla mcts with 10k rollout.

## Performance 
I don't make precise measurement of perfromance.

Just to give  an idea: my configuration is Core7, 16G RAM and Nvdia 1070.

One iteration on 4IARow: 400 rollouts, 64x4 net, 1000 games per iteration takes about 6 to 8 minutes with 200 games in parallel, training time included. With a 64x7 nets this time goes up to 12 minutes.

Compared to implementation in Python this is orders of magnitude faster.
