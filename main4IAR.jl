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



include("Games/4IARow//4IAR.jl")
 include("Games/4IARow/4IARnet.jl")
 include("mctsworkers.jl")
# include("metamcts.jl")
 include("selfplay.jl")
 include("train.jl")
#
# function startFromScratch()
#     #@load "Games/Reversi/Data/reseau21.json" reseau
#     #actor=reseau|>gpu
#     trainingPipeline(200,100,actor)
# end
