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
end

function nodeInit(parent::Union{Nothing,Node}, actionFromParent::Int, state::GameEnv)
    n = getActionNumber(state)
    actions = Vector{Action}([Action() for k = 1:n])
    return Node(parent, actionFromParent, state, false,false, 0, actions, Dict())
end


function descendTree(root::Node, treePolicy::String, c::Float32, ω::Float32)
    current = root
    while current.expanded
        addVisit(current)
        best = bestChild(current, treePolicy, c, ω)
        addVisitAction(current, best)
        current = maybeAddChild(current, best)
    end
    addVisit(current)
    return current
end

function expand(leaf::Node, treePolicy::String, treshhold::Int)

    if !isTerminal(leaf) #&& leaf.visits >= treshhold
        leaf.expanded = true
    end

end

function evaluate(leaf::Node, treePolicy::String, eval = nothing)::Float32
    if treePolicy == "classic"
        v = simul(leaf.state)

    elseif treePolicy == "puct"
        #w=@spawn simul(leaf.state,1) ###########"if simulation is used for value"
            if isTerminal(leaf)
                return simul(leaf.state)
            else
                p,v=eval(leaf.state)
                for k in 1:getActionNumber(leaf)
                    leaf.actions[k].prior=p[k]
                end

                return v[1]
            end
    else
        print("this policy is unknown")
    end
    return v
end
function discreteprob(x)
    if x<-1/3
        return Float32(-1)
    elseif x>1/3
        return Float32(1)
    else
        return Float32(0)
    end
end
function evaluate(leaf::Vector{Node}, treePolicy::String,dirichPortion,dirichAlpha, eval = nothing)
    L=length(leaf)
    temp = zeros(sizeInput..., L) ###### hack for dimension auto
    I=CartesianIndices(sizeInput)
    #v= zeros(Float32,L)
   @threads for i = 1: L# enlevé mémoire ?
        temp[I, i] .= decoder(leaf[i].state)[I, 1]
        #v[i]=simul(leaf[i].state,100)                              ####### if you use simulation for value ###
    end
    p,v= (eval(temp |> gpu) |> cpu)
   for i = 1:L
        current = leaf[i]
        if isRoot(current)
            nA=getActionNumber(current)
            dir=rand(Dirichlet(nA,10.8/nA))
            normalize=0
            lp=getNumber.(legalPlays(current.state)) ### attention legal plays modifié####
            prior=softmax(p[lp,i])
            #prior=sum(prior)==0 ? ones(nA)/nA : prior/sum(prior)
            for j in 1:nA
                current.actions[j].prior=(1-dirichPortion)*prior[j]+dirichPortion*dir[j]# p[getNumber(findIndex(current.state, j)),i]
            end
            # prior=softmax(prior)
            # for j in 1:nA
            #     current.actions[j].prior=prior[j]
            # end

            # mx=maximum([current.actions[j].prior for j in 1:nA])
            # if mx>0.65 && nA>1                                     ##### hack against overfitting: limit probality concentration during utcsearch
            #     λ=(0.65-1/nA)/(mx-1/nA)                            ######  maybe not good
            #     for j in 1:nA
            #         current.actions[j].prior=λ*current.actions[j].prior+(1-λ)/nA
            #     end
            # end

            # for j in 1:nA
            #     current.actions[j].prior = current.actions[j].prior+dirichPortion*dir[j]
            # end



        else

            nA=getActionNumber(current)
            if isTerminal(current)
                v[i]=simul(current.state)
                continue
            else
                if nA==0
                    print(current.state)
                end
                lp=getNumber.(legalPlays(current.state))
                prior=softmax(p[lp,i])
                #prior=sum(prior)==0 ? ones(nA)/nA : prior/sum(prior)
                for j in 1:nA
                    current.actions[j].prior= prior[j]#p[getNumber(findIndex(current.state, j)),i]
                end
            end
            # prior=softmax(prior)
            # for j in 1:nA
            #     current.actions[j].prior=prior[j]
            # end
        end

    end
    return v
    #return discreteprob.(v)

end

function backUp(leaf::Node, v::Float32,type)
    current = leaf.parent
    move = leaf.actionFromParent
    while current != nothing
        addw(current,move,1+(1-v)/2)
        current.actions[move].n-=2
        move = current.actionFromParent
        current = current.parent
        v = -v
    end
end

################# Utility ########################
function getActionNumber(node::Node)
    return getActionNumber(node.state)
end

function bestChild(node::Node, treePolicy::String, c::Float32, ω::Float32)
    best = -1
    move = -1
    if treePolicy== "classic"
        for (k, action) in enumerate(node.actions)
            if action.n == 0
                v = Inf
            else
                v = action.w / action.n + c * sqrt(log(node.visits) / action.n)
            end
            if v > best
                best = v
                move = k
            end
        end
        return move
    elseif treePolicy == "puct"
        move=[]
        ##### pour fpu leela zero et changer exploration à la racine
        if isRoot(node)
            cp=c
            cfpu=0
        else
            cp=c
            cfpu=0.2
        end
        ###############MCTS for deterministic games###########
        # for action in node.actions
        #     if action.w>0.9*action.n
        #         cp=0
        #         break
        #     end
        # end
        ####################################
        vnode=sum(action.w for action in node.actions)
        dvnode=sum(action.n for action in node.actions)
        if dvnode!=0
            vnode=vnode/dvnode
            reduction=sqrt(max(0,sum(action.prior for action in node.actions if action.n>0)))
        else
            vndode=0
            reduction=0
        end

        for (k, action) in enumerate(node.actions)

            v = action.w / (action.n+1) + cp*action.prior* sqrt(node.visits) / (action.n+1)#+(vnode-cfpu*reduction)*(action.n==0) ##### fpu leela zero

            if v > best
                best = v
                move = [k]
            elseif v==best
                push!(move,k)
            end
        end
        if move==-1
            println("winner", winner(node.state))
            println(getActionNumber(node.state))
            print(node.state)
            println(node.visits, node.actions)
        end
        return rand(move)
    else
        print("policy unknown")
    end

end

function addVisit(node::Node)
    node.visits += 1
end

function addVisitAction(node::Node, action::Int)
    node.actions[action].n += 3
    node.actions[action].w-=1
end

function maybeAddChild(node::Node, move::Int)
    current = node
    if !haskey(current.child, move)
        current = playIndex(current, move)
        node.child[move] = current
    else
        current = current.child[move]
    end
    return current
end

function playIndex(node::Node, move::Int)
    state = playIndex(node.state, findIndex(node.state,move))
    return nodeInit(node, move, state)
end

function isTerminal(node::Node)
    return winner(node.state)[1]
end

function isRoot(node::Node)
    return node.parent==nothing
end

function addw(node::Node, move::Int, value::Float32)
    node.actions[move].w += value
end

function setPrior(node::Node, p::Vector{Float32})
    for (k, action) in enumerate(node.actions)
        action.prior = p[k]
    end
end


################################## mcts rootine#############""
function rollout(###################@thredas enlevé partout mé moire ?
    root::Vector{Node},
    treePolicy::String="puct",
    dirichPortion::Float32=0.25,
    dirichAlpha::Float32=1,
    treshhold::Int=0,
    c::Float32=1.1,
    ω::Float32=0,
    eval = nothing,
    workers=1
)
    L=length(root)

    leaf=Vector{Node}(undef,L*workers)

     @threads for i in 1:L
        for j in 1:workers
            leaf[workers*(i-1)+j] = descendTree(root[i], treePolicy, c, ω)
            expand(leaf[workers*(i-1)+j], treePolicy, treshhold)
        end
    end
    v = evaluate(leaf, treePolicy,dirichPortion,dirichAlpha, eval)
    @threads     for i in 1:L
        for j in 1:workers
            backUp(leaf[workers*(i-1)+j], v[workers*(i-1)+j],treePolicy)
        end
    end
end

function rollout(
    root::Node,
    treePolicy::String,
    treshhold::Int,
    c::Float32,
    ω::Float32,
    eval = nothing,
)
    leaf = descendTree(root, treePolicy, c, ω)
    expand(leaf, treePolicy, treshhold)
    v = evaluate(leaf, treePolicy, eval)
    backUp(leaf, v,treePolicy)
end

struct TreePolicy
    type::String
    dirichNoise::Bool
    dirichPortion::Float32
    dirichα::Float32
    treshhold::Int
    c::Float32
    ω::Float32
    eval
    workers::Int
end

function (m::TreePolicy)(root::Node, readout::Int)
    for cpt = 1:readout
        rollout(root, m.type, m.treshhold, m.c, m.ω, m.eval)
    end
    return extractRoot(root,m.type)
end

function (m::TreePolicy)(root::Vector{Node}, readout::Int,test=false)
    while root[1].visits<readout
        rollout(root, m.type, m.dirichPortion,m.dirichα,m.treshhold, m.c, m.ω, m.eval,m.workers)
    end
    if test
        return root
    else
        return extractRoot.(root,m.type)
    end
end

(m::TreePolicy)(game::Vector{GameEnv}, readout::Int) =
    m([nodeInit(nothing, 0, g) for g in game], readout)

(m::TreePolicy)(game::GameEnv, readout::Int,test=false) =
        m([nodeInit(nothing, 0, game)], readout,test)[1]


function extractRoot(node::Node,type="puct")
        n=getActionNumber(node)
        p=zeros(maxActions) #############hack for extractroot
        for k in 1:n
            p[getNumber(findIndex(node.state,k))]=node.actions[k].n/node.visits
        end
        return p,2*sum(action.w for action in node.actions)/node.visits-1
end

function vanilla(game,readout)
    root=nodeInit(nothing, 0, game)
    for cpt = 1:readout
        rollout(root, "classic",0,1.2f0, 0.0f0, nothing)
        end

    return extractRoot(root,"puct")
end
