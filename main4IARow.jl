

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

include("4IARow.jl")
include("DenseNet.jl")
include("mcts_gpu_regulnewt.jl")
include("mctsthreaded_rev.jl")
include("selfplay_rev.jl")
include("train.jl")
