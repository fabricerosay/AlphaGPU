
using CUDA

using Base.Iterators: partition
using Juno: @progress
using Statistics
using Flux
using Flux:@functor
using Distributions
using Random
using StaticArrays
#using StatsBase
using Zygote
using Base.Threads:@threads,@spawn
using JLD2
using LinearAlgebra
using DataStructures
#using ProgressMeter
#using Gadfly
using ArgParse


const N=7
const NN=N*N
const UnrollSteps=5

include("Bitboard.jl")
include("Hex.jl")

module Game
    export Position, canPlay,play,isOver,affiche,VectorizedState,FeatureSize,maxActions,maxLengthGame,PoolSample,push_buffer,update_buffer,length_buffer
    using ..Hex
    mutable struct Sample
        state::Vector{Int8}
        policy::Array{Float32,2}
        move::Vector{Int}
        player::Int8
        value::Float32
        fstate::Vector{Int8}
    end

    Sample(k::Int)=Sample(zeros(Int8,2*VectorizedState),zeros(Float32,(maxActions,k+1)),[0 for j in 1:k],1,0,zeros(Int8,FeatureSize))


    mutable struct PoolSample
        length::Int
        currentIndex::Int
        pool::Vector{Sample}
        full::Bool
        unrollsteps::Int
    end

    PoolSample(N::Int,k::Int)=PoolSample(N,1,Sample[Sample(k) for j in 1:N],false,k)

    function push_buffer(buffer::PoolSample,state,policy,player,i)
        index=buffer.currentIndex
        @views begin
            buffer.pool[index].state.=state[:,i]
            buffer.pool[index].policy[:,1].=policy[i,:]
        end
        buffer.pool[index].player=player

        newindex=index==buffer.length ? 1 : index+1
        if newindex==1
            buffer.full=true
        end
        buffer.currentIndex=newindex
        return index
    end

    function update_buffer(buffer::PoolSample,index,result,fstate)
        L=length(index)
        unrollsteps=buffer.unrollsteps
        for (k,id) in enumerate(index)
            player=buffer.pool[id].player
            buffer.pool[id].value=(1+result*player)/2
            buffer.pool[id].fstate.=fstate*player
            for i in 1:unrollsteps
                j=min(L,k+i)
                  buffer.pool[id].policy[:,i+1].=buffer.pool[index[j]].policy[:,1]
                buffer.pool[id].move[i]=buffer.pool[index[j-1]].move[1]
            end
        end
    end

    length_buffer(buffer::PoolSample)=buffer.full ? buffer.length : buffer.currentIndex-1
end
using .Game

include("DenseNet.jl")
include("mcts_gpu.jl")
include("selfplay.jl")
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
    default = 1.5f0
    "--noise"
    help = "uniform noise at the root, default to 1/maxActions"
    arg_type = Float32
    default = Float32(2/Game.maxActions)
end

parsed_args = parse_args(ARGS, s)



function main(generation)
     #JLD2.@load "DataHex/reseau400.json" reseau
     #net=reseau|>gpu
    net=MuNet(2*Game.VectorizedState,Game.maxActions,Game.FeatureSize,512,4,4)|>gpu
    #net=ressimplesf(2*Game.VectorizedState,Game.maxActions,Game.FeatureSize,512,4)|>gpu
    trainingnet=deepcopy(net)
    buffer=PoolSample(2000000,UnrollSteps)
    best=1
    currentelo=-1000
    for i in 1:generation
        # buffer=trainingPipeline(net,trainingnet,buffer,i,currentelo,game="Hex",cpuct=parsed_args["cpuct"],noise=parsed_args["noise"],samplesNumber=parsed_args["samples"],rollout=parsed_args["rollout"],
        # iteration=1,batchsize=parsed_args["batchsize"],sizein=2*Game.VectorizedState,sizeout=Game.maxActions,fsize=Game.FeatureSize)
        net,trainingnet,passing,currentelo=trainingPipeline(net,trainingnet,buffer,i,currentelo,game="Hex",cpuct=parsed_args["cpuct"],noise=parsed_args["noise"],samplesNumber=parsed_args["samples"],rollout=parsed_args["rollout"],
        iteration=1,batchsize=parsed_args["batchsize"],sizein=2*Game.VectorizedState,sizeout=Game.maxActions,fsize=Game.FeatureSize)
        reseau=net|>cpu

        if passing
            best=i
        end
        println("meilleur réseau: $best")
        println("elo actuel: $currentelo, generation: $i")


        net=reseau|>gpu
    end
end

main(parsed_args["generation"])
