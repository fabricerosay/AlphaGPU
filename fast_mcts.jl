module FMCTS

using CUDA
using Flux
using StatsBase
using Distributions
using StaticArrays
using Base.Threads:@threads
import ..Game:play
using ..Game
#using ..Game:Position
export MctsContext

mutable struct Action
    w::Float32
    n::Float32
    prior::Float32
end

Action( prior = 0) = Action( 0, 0, prior)

mutable struct Node
    parent::Union{Nothing,Node}
    actionFromParent::Int
    state::Position
    expanded::Bool
    visits::Int
    actions::Vector{Action}
    child::Dict{Int,Node}
end

function getActionNumber(pos::Position)
    A=0
    for k in 1:maxActions
        if canPlay(pos,k)
            A+=1
        end
    end
    return A
end

function newton(actions,λ)
	α=0f0
	for action in actions
		gap=max(λ*action.prior,1f-4)
		α=max(α,action.n==0 ? gap : action.w/(action.n)+gap)
	end
	err=Inf32
	newerr=Inf32
	for j in 1:100
		S=0f0
		g=0f0
		for action in actions
			top=λ*action.prior
			bot=action.n==0 ? α : (α-action.w/(action.n))
			S += top/bot
			g += -top/(bot*bot)
		end

		newerr = S - 1f0

		if newerr<0.001f0 || newerr==err
			break
		else
			α -= newerr/g
			err= newerr
		end
	end
	return α
end

function nodeInit(parent::Union{Nothing,Node}, actionFromParent::Int, state::Position)
    actions = Vector{Action}([Action() for k = 1:maxActions])
    return Node(parent, actionFromParent, state, false, 0, actions, Dict())
end


function descendTree(root::Node, c::Float32)
    current = root
    while current.expanded

        addVisit(current)

        best = bestChild(current, c)

        addVisitAction(current, best)

        current = maybeAddChild(current, best)

    end

    addVisit(current)

    return current
end

function expand(leaf::Node, prior)
        leaf.expanded = true
        normalize=0
        for j in 1:maxActions
            if canPlay(leaf.state,j)
                leaf.actions[j].prior= prior[j]
                normalize+=prior[j]
            end
        end
        for j in 1:maxActions
                leaf.actions[j].prior/= normalize
        end
end

function decode_cpu(pos,fstate=nothing)
    if fstate==nothing
        fstate=zeros(Int8,2*VectorizedState)
    end
    bplayer=pos.bplayer
    bopponent=pos.bopponent
    #zone=pos.zone
    for j in 1:VectorizedState
        if bplayer[j]
            fstate[j]=1
        else
            fstate[j]=0
        end
        if bopponent[j]
            fstate[j+VectorizedState]=1
        else
            fstate[j+VectorizedState]=0
        end

    end
    # for j in 1:9
    #     if j==zone
    #         fstate[j+180]=1
    #     else
    #         fstate[j+180]=0
    #     end
    # end
    return fstate
end

function evaluate(leaf::Node,actor,prealloc,komi=0)
    f,r=isOver(leaf.state)
    if f
        if r>0
            r=1
        elseif r<0
            r=-1
        end

        return true,zeros(Float32,maxActions),(1+r*leaf.state.player)/2
    else
         p,v=actor(decode_cpu(leaf.state,prealloc))#,[komi])
         return false,p,v[1]
    end
end


function backUp(leaf::Node, v)
    current = leaf.parent
    move = leaf.actionFromParent

    while current != nothing

        addw(current,move,(1-v))

        move = current.actionFromParent
        current = current.parent
        v=1-v
    end
end


################# Utility ########################
function getActionNumber(node::Node)
    return Game.getActionNumber(node.state)
end

# function bestChild(node::Node, c::Float32)
#     best = -Inf
#     move=-1
#
#     if isRoot(node)
#         cp=c
#         cfpu=0
#     else
#         cp=c
#         cfpu=0.2
#     end
#
#     vnode=sum(action.w for action in node.actions)/node.visits
#
#     reduction=sqrt(sum(action.prior*(action.n>0) for action in node.actions))
#
#
#     for (k, action) in enumerate(node.actions)
#
#         ac=action.w/(action.n+1)#+(vnode-cfpu*reduction)*(action.n==0)
#
#         v = ac + cp*action.prior* sqrt(node.visits) / (action.n+1)
#
#         if v > best
#             best = v
#             move = k
#         end
#
#     end
#
#     return move
#
#
# end

function bestChild(node::Node, c::Float32)
    λ=c*sqrt(node.visits)/(getActionNumber(node.state)+node.visits)
    α=newton(node.actions,λ)
    π=[action.n==0 ?  λ*action.prior/α : λ*action.prior/(α-action.w/(action.n)) for action in node.actions]
    return sample(1:maxActions,Weights(π))
end

function addVisit(node::Node)

    node.visits += 1

end

function addVisitAction(node::Node, action::Int)

    node.actions[action].n += 1

end

function maybeAddChild(node::Node, move::Int)
    current = node
    if !haskey(current.child, move)
        current = play(current, move)

        node.child[move] = current

    else
        current = current.child[move]
    end
    return current
end

function play(node::Node, move::Int)
    state = play(node.state, move)
    return nodeInit(node, move, state)
end

function isTerminal(node::Node)
    return winner(node.state)[1]
end

function isRoot(node::Node)
    return node.parent==nothing
end

function addw(node::Node, move::Int, value)

        node.actions[move].w += value
end



struct MctsContext
    c::Float32
    nn
    prealloc
end



function (ctx::MctsContext)(pos::Position,readout,komi=0)

    node=nodeInit(nothing,-1,pos)
    for cpt in 1:readout
        leaf=descendTree(node,ctx.c)
        back,p,v=evaluate(leaf,ctx.nn,ctx.prealloc,komi*leaf.state.player*pos.player)
        if back
            backUp(leaf,Float32(v))
            continue
        else
            expand(leaf,p)
            backUp(leaf,v)
        end

    end



    return extractRoot(node,ctx.c)

end



function extractRoot(node::Node,c)

        N=node.visits
        #p=[action.n/N for action in node.actions]
        λ=c*sqrt(node.visits)/(getActionNumber(node.state)+node.visits)
        α=newton(node.actions,λ)
        p=[action.n==0 ?  λ*action.prior/α : λ*action.prior/(α-action.w/(action.n)) for action in node.actions]
        return p,sum(action.w for action in node.actions)/N

end

end
