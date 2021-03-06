# AlphaGPU
Alphazero on GPU thanks to CUDA.jl

This is another implementation of alphazero algorithm, where almost everything happens on the gpu like in https://arxiv.org/abs/2104.03113.
The core algorithm is Alphazero modified as explained in https://arxiv.org/abs/2007.12509
The implementation works for Gobang( any size up to 13x13, tested up to 9x9), 4 in a row, reversi 6x6, reversi 8x8 and Hex any size up to 13x13 (tested up to 9x9).

It's more an exercice to learn programming CUDA in julia than a serious implementation of alphazero.

Few technical details:

-First clone the repository.

-Right now the code is messy and here just to showcase the possibility of CUDA.jl

-If you want to tweek things you'll have to dig into the code... take a deep breath, I'll be happy to answer any questions( for example right now the size and depth of the network is fixed, but can be manually chnaged in the script mainNameoftheGame.jl, the size is 128 for hidden layers and there are 6 of those, for bigger board than tic tac toe i'll recommand 512x8)

-You can change the board size for Gobang (tested up to 9x9) and Hex , for that edit the file mainGoBang.jl and change N to whatever you want(less than 13) and Nvict (for GoBang only, number of aligned pieces to win).
-By default N=Nvict=3 : Tic Tac Toe.

-Once you have selected the size you want you can launch the training (for Hex) with the following commande line:

julia --project --math-mode=fast --check-bounds=no  mainHex.jl --cpuct 1.5  --rollout 64 --samples 32768 --generation 50

usage: main4IARow.jl [--samples SAMPLES] [--rollout ROLLOUT]
                     [--generation GENERATION] [--batchsize BATCHSIZE]
                     [--cpuct CPUCT] [--noise NOISE] [-h]

optional arguments:
  --samples SAMPLES     number of selfplay games per generation (type:
                        Int64, default: 32768)
  --rollout ROLLOUT     number of rollouts (type: Int64, default: 64)
  --generation GENERATION
                        number of generations (type: Int64, default:
                        100)
  --batchsize BATCHSIZE
                        batchsize for training (type: Int64, default:
                        8192)
  --cpuct CPUCT         cpuct (exploration coefficient in cpuct
                        formula) (type: Float32, default: 2.0)
  --noise NOISE         uniform noise at the root, default to
                        1/maxActions (type: Float32, default:
                        0.142857)
  -h, --help            show this help message and exit

Depending on the memory of you GPU and the size of the board, you will have to reduce the number of samples. For Tic Tac Toe 20 generations are enough to get a good network. Right now the exploration is too high for Tic Tac Toe, so the training could be faster, but it works well for 9x9.

# Results
-Up to 8x8 the network is able to draw Embryo after a few hours of training (4 or 5 I think, probably less with this version)

-For Tic TacToe it plays probably perfectly after a few minutes

-For 9x9 and five in row, it sometimes draw Embryo and sometimes it makes huge Blunder, but the training time is longer to get there (around 10 hours probably).

-The same algorithm can produce very strong nets in 30 minutes for Connect4

-For Hex on 7x7 board, net is able to beat Mohex when starting, after 35 minutes. You can manually try a trained net using the script testHex.jl.
To Lauch a game you need Luxor.jl and commande is testvsordi(net,readout,firstplayer), firstplayer=1 net starts, firstplayer=-1, net goes second.

-For Reversi things are less convincing but after 2 or 3 hours the net is able to draw or win perfect player when going second( but it seems far from being perfect)


-For speed noise is uniform and should be set to 1/average number of moves, then you should play around this value. Usually if loss is dropping to fast you should set noise higher. With so few rollouts as 64, and loss getting too low, the net will be caught in local optima too early.

# Implementing your own game
-Take a look at Gobang.jl to get hints.

-All you have to do is a struct Position, the functions canPlay, play and isOver.

-The only problematic part is the vectorized state encoding wich is not at all generic atm. It will maybe need a little work (again don't hesitate). It will work out of the bag if you have one bitboard for player one and one bitboard for player 2 in your struct position (both of size VectorState see mcts_gpu_gobang.jl and Gobang.jl for details)

-What is mandatory is the struct to be non mutable. Also instead of using arrays as is customary, you will have to use bitboard from Bitbard.jl which are nothing but a replica of BitArray, but non mutable, with the addition of few helpfull functions like bit rotation. It's far from perfect and not at all generic atm.

-As bitboard are not mutable you'll have to use setindex(bb,value,index) instead of bb[index]=value, but you can get the value with bb[index] (value is true or false), and you can also use bb[x,y] for 2 dimentionnal bitboards wich is just a convenience (bitboard are nothing but 3 UInt64 wrapped in a nicer struct)

-using Those you are not restricted to code only the board, of the players but can add features (for example you could code in the state the places where you loose or win immediately, it should make training faster but didn't try)

-It's very easy to code 4 in row that way, Reversi and Ultimate Tic Tac Toe (soon to be added), and probably checkers but never tried.

# Where to go from here
-optimize the code further (right now the major bottleneck is the policy optimization step during mcts)

-add katago or muzero enhancement(on the way)

-change the policy update(mpo, i.e. \pi*exp(adv/\lambda)) style is very effective at test time, less convincing at training time the idea is that puct converged to a regularized policy and you could change the regularization making it closer to mpo or ppo update. The muesli paper is very tempting...)
