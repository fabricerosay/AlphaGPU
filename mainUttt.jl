using Flux
using Flux: @epochs, onehotbatch, mse, throttle,@functor,kldivergence
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



include("Games/UTTT/Uttt.jl")
include("Games/UTTT/UtttNet.jl")
include("mctsworkers.jl")
include("selfplay.jl")
include("train.jl")

function startFromScratch()
    #@load "Games/Reversi/Data/reseau21.json" reseau
    #actor=reseau|>gpu
    trainingPipeline(100,100,actor,r;game="UTTT")
end
###### need hack for autodimension in selfplay et mcts evaluate
######## number action extractroot

####### params bon run  trainingPipeline(actor;nbworkers=100,game="UTTT",bufferSize=100000,samplesNumber=500,
#####       rollout=200,iteration=100,cpuct=2,chkfrequency=1,batchsize=256,lr=0.001,epoch=1,temptresh=10)
