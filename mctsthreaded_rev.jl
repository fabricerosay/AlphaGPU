using Base.Threads


mutable struct Action
    target::Union{Nothing,Int}
    w::Float32
    n::Float32
    prior::Float32
end

Action(target = nothing, prior = 0) = Action(target, 0, 0, prior)

mutable struct Node
    parent::Union{Nothing,Node}
    actionFromParent::Int
    state::GameEnv
    expanded::Bool
    prioritized::Bool
    visits::Int
    actions::Vector{Action}
    child::Dict{Int,Node}
    lck::ReentrantLock
end

function nodeInit(parent::Union{Nothing,Node}, actionFromParent::Int, state::GameEnv)
    n = getActionNumber(state)
    actions = Vector{Action}([Action() for k = 1:n])
    return Node(parent, actionFromParent, state, false,false, 0, actions, Dict(),ReentrantLock())
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

function expand(leaf::Node, prior,dirichPortion)
    lock(leaf.lck)
        leaf.expanded = true
        if isRoot(leaf)
            nA=getActionNumber(leaf)
            dir=rand(Dirichlet(nA,1))
            lp=getNumber.(legalPlays(leaf.state)) ###
            prior=prior[lp]
            prior=prior/sum(prior)
            for j in 1:nA
                leaf.actions[j].prior=(1-dirichPortion)*prior[j]+dirichPortion*dir[j]# p[getNumber(findIndex(leaf.state, j)),i]
            end

        else

            nA=getActionNumber(leaf)


                lp=getNumber.(legalPlays(leaf.state))
                prior=prior[lp]
                prior=prior/sum(prior)


                for j in 1:nA
                    leaf.actions[j].prior= prior[j]
                end

        end
    unlock(leaf.lck)
end

function evaluate(leaf::Node,actor)
        if isTerminal(leaf)
            return nothing,simul(leaf.state)
        else
            return actor([leaf.state])[1]
        end
end


function backUp(leaf::Node, v)
    current = leaf.parent
    move = leaf.actionFromParent

    while current != nothing

        addw(current,move,(1-v)/2)

        move = current.actionFromParent
        current = current.parent
        v=-v
    end
end

function revert(leaf::Node)
    lock(leaf.lck)
        leaf.visits-=1
    unlock(leaf.lck)
    current = leaf.parent
    move = leaf.actionFromParent

    while current != nothing
        lock(current.lck)
        current.actions[move].n-=1
        current.visits-=1
        unlock(current.lck)
        move = current.actionFromParent
        current = current.parent
    end
end
################# Utility ########################
function getActionNumber(node::Node)
    return Game.getActionNumber(node.state)
end

function bestChild(node::Node, c::Float32)
    best = -Inf
    move=[]

    if isRoot(node)
        cp=c
        cfpu=0
    else
        cp=c
        cfpu=0.2
    end

    vnode=sum(action.w for action in node.actions)/node.visits

    reduction=sqrt(sum(action.prior*(action.n>0) for action in node.actions))


    for (k, action) in enumerate(node.actions)

        ac=action.w/(action.n+1)#+(vnode-cfpu*reduction)*(action.n==0)

        v = ac + cp*action.prior* sqrt(node.visits) / (action.n+1)

        if v > best
            best = v
            move = k
        end

    end

    return move


end

function addVisit(node::Node)
    lock(node.lck)
        node.visits += 1
    unlock(node.lck)
end

function addVisitAction(node::Node, action::Int)
    lock(node.lck)
    node.actions[action].n += 1

    unlock(node.lck)
end

function maybeAddChild(node::Node, move::Int)
    current = node
    lock(node.lck)
    if !haskey(current.child, move)
        current = playIndex(current, move)

        node.child[move] = current

    else
        current = current.child[move]
    end
    unlock(node.lck)
    return current
end

function playIndex(node::Node, move::Int)
    state = Game.playIndex(node.state, findIndex(node.state,move))
    return nodeInit(node, move, state)
end

function isTerminal(node::Node)
    return winner(node.state)[1]
end

function isRoot(node::Node)
    return node.parent==nothing
end

function addw(node::Node, move::Int, value)
    lock(node.lck)
        node.actions[move].w += value
    unlock(node.lck)
end



struct MctsContext
    dirichPortion::Float32
    c::Float32
    nworkers::Int
    nn
end



function (ctx::MctsContext)(roots::Vector{Node},readout,value=false)
    RN=length(roots)
    copyroots=[root for root in roots]
    leaves=Vector{Node}(undef,RN*ctx.nworkers)
    batch=Vector{Node}()
    collisions=0
    cnt_eval=0
    while !isempty(roots)
        @threads for i in 1:length(roots)
            @threads for j in 1:ctx.nworkers
                leaves[(i-1)*ctx.nworkers+j]=descendTree(roots[i],ctx.c)

            end
        end
        for i in 1:length(roots)
            for j in 1:ctx.nworkers
                leaf=leaves[(i-1)*ctx.nworkers+j]
                if isTerminal(leaf)
                    f,r=winner(leaf.state)
                    player=leaf.state.player==1 ? 1 : -1
                    backUp(leaf,Float32(r*player))
                else
                    if leaf.prioritized
                        revert(leaf)

                    else
                        leaf.prioritized=true
                        push!(batch,leaf)
                    end
                end
            end

        end
        if !isempty(batch)
            if Sys.free_memory() / 2^20 < 2000
                println("memory reclaim")

            end
            if cnt_eval>readout
                GC.gc(true)
                CUDA.reclaim()
                CUDA.reclaim()
                cnt_eval=0
            end
            cnt_eval+=1

            π,v=ctx.nn(([b.state for b in batch]))

             @threads for i in eachindex(batch)
                @views expand(batch[i],π[:,i],ctx.dirichPortion)
                backUp(batch[i],v[1,i])
            end
            empty!(batch)
        end
        for (i,root) in enumerate(roots)
            if root.visits>=readout
                deleteat!(roots,i)
            end
        end
    end

    GC.gc()

    if value
        return extractRoot(copyroots[1],true)
    else
        return extractRoot.(copyroots)
    end
end

(m::MctsContext)(games::Vector{GameEnv},n)=m([nodeInit(nothing,-1,game) for game in games],n)

function extractRoot(node::Node,value=false)
        n=getActionNumber(node)
        p=zeros(maxActions)
        q=zeros(maxActions)
        N=node.visits
       
       
       
        for k in 1:n
            p[getNumber(findIndex(node.state,k))]=node.actions[k].n/node.visits
        end
        if value
            return p,sum(action.w for action in node.actions)/node.visits
        else
            return p
        end
end
