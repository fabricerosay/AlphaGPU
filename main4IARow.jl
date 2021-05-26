

using CUDA

using Base.Iterators: partition
using Juno: @progress
using Statistics
using Flux
using Flux:@functor
using Distributions
using Random
using StaticArrays
using StatsBase
using Zygote
using Base.Threads:@threads,@spawn
using JLD2
using LinearAlgebra
using DataStructures
using ProgressMeter
using Gadfly
using ArgParse

include("Bitboard.jl")
include("4IARow.jl")
module Game
    export Position, canPlay,play,isOver,affiche,VectorizedState,maxActions,maxLengthGame
    using ..FourIARow
end
include("DenseNet.jl")
include("mcts_gpu.jl")
include("selfplay_rev.jl")
include("train.jl")



s = ArgParseSettings()
@add_arg_table! s begin
    "--samples"
    help = "number of selfplay games per generation"
    arg_type = Int
    default = 32*1024
    "--rollout"
    help = "number of rollouts"
    arg_type = Int
    default = 64
    "--generation"
    help = "number of generations"
    arg_type = Int
    default = 100
    "--batchsize"
    help = "batchsize for training"
    arg_type = Int
    default = 2*4096
    "--cpuct"
    help = "cpuct (exploration coefficient in cpuct formula)"
    arg_type = Float32
    default = 2f0
    "--noise"
    help = "uniform noise at the root, default to 1/maxActions"
    arg_type = Float32
    default = Float32(1/Game.maxActions)
end

parsed_args = parse_args(ARGS, s)

actor=ressimples(2*FourIARow.VectorizedState,FourIARow.maxActions,512,6)|>gpu

trainingPipeline(actor,game="4IARow",cpuct=parsed_args["cpuct"],noise=parsed_args["noise"],samplesNumber=parsed_args["samples"],rollout=parsed_args["rollout"],
iteration=parsed_args["generation"],batchsize=parsed_args["batchsize"],sizein=2*FourIARow.VectorizedState,sizeout=FourIARow.maxActions)
