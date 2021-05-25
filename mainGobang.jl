

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
#include("Gobang.jl")
include("DenseNet.jl")
#include("mcts_gpu_gobang_q.jl")
include("selfplay_rev.jl")
include("train.jl")



# s = ArgParseSettings()
# @add_arg_table! s begin
#     "--samples"
#     help = "number of selfplay games per generation"
#     arg_type = Int
#     default = 32*1024
#     "--rollout"
#     help = "number of rollouts"
#     arg_type = Int
#     default = 64
#     "--generation"
#     help = "number of generations"
#     arg_type = Int
#     default = 100
#     "--batchsize"
#     help = "batchsize for training"
#     arg_type = Int
#     default = 2*4096
# end
#
# parsed_args = parse_args(ARGS, s)
#
# actor=ressimplesq(2*GoBang.VectorizedState,GoBang.maxActions,128,6)|>gpu
#
# trainingPipeline(actor,samplesNumber=parsed_args["samples"],rollout=parsed_args["rollout"],
# iteration=parsed_args["generation"],batchsize=parsed_args["batchsize"],sizein=2*GoBang.VectorizedState,sizeout=GoBang.maxActions)
