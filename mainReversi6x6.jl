

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
include("Reversi6x6.jl")
module Game
    export Position, canPlay,play,isOver,affiche,VectorizedState,maxActions,maxLengthGame
    using ..RevSix
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
end

parsed_args = parse_args(ARGS, s)

actor=ressimples(2*RevSix.VectorizedState,RevSix.maxActions,512,6)|>gpu

trainingPipeline(actor,game="Reversi",samplesNumber=parsed_args["samples"],rollout=parsed_args["rollout"],
iteration=parsed_args["generation"],batchsize=parsed_args["batchsize"],sizein=2*RevSix.VectorizedState,sizeout=RevSix.maxActions)