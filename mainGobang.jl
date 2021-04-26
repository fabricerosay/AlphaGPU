using Flux
using Flux: @epochs, onehotbatch, mse, throttle,@functor#,GlobalMaxPool
using Base.Iterators: partition
using Juno: @progress
using Statistics
using CuArrays
using Distributions
using Random
using StaticArrays
using StatsBase
using Zygote
using Base.Threads:@threads,@spawn
using BSON: @save, @load
using LinearAlgebra


include("Games/Gobang/Gobang.jl")
include("Games/Gobang/GobangNet.jl")
include("mctsworkers.jl")
# include("metamctsStandardDeviation.jl")
include("selfplay.jl")
include("train.jl")

function startFromScratch(actor)
    #@load "Games/Reversi/Data/reseau21.json" reseau
    #actor=reseau|>gpu
    trainingPipeline(actor;nbworkers=100,game="Gobang",bufferSize=100000,samplesNumber=100,
    rollout=100,iteration=100,chkfrequency=1,batchsize=64,lr=0.001,epoch=1,temptresh=10)
end
###### need hack for autodimension in selfplay et mcts evaluate
######## number action extractroot


######## Fall back résultat et value donnée par mcts après 10 itérations sinon classique
