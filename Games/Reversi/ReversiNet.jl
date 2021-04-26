#####################all network output logits###############
const sizeInput=(6,6,3)


mutable struct resnet #########residual block##############
    c1
    b1
    c2
    b2
end
@functor resnet

resnet(n_filter)=resnet(Conv((1,1),n_filter=>n_filter,pad=0),BatchNorm(n_filter),Conv((3,3),n_filter=>n_filter,pad=1),BatchNorm(n_filter,relu))

 (m::resnet)(x)=m.b1(m.c1(x)+m.c2(x))

mutable struct resconv ############residual policy only####################
    base
    res
    policy
end

mutable struct resconv2heads ##################residual policy and value###############
    base
    res
    policy
    value
end

Flux.@functor resconv
Flux.@functor resconv2heads


function (m::resconv)(x)
    b=m.base(x)
    for r in m.res
        b=r(b)
    end
    return (m.policy(b))
end

function (m::resconv2heads)(x)
    b=m.base(x)
    for r in m.res
        b=r(b)
    end
    #b=reshape(b,:,size(b,4))
    return (m.policy(b),m.value(b))
end

function resconv(n_filter,n_tower) ############### n_filter: number of filter, n_tower number of residual block#######

     return resconv(Chain(Conv((3,3),12=>n_filter,pad=1),BatchNorm(n_filter,relu)),[resnet(n_filter) for k in 1:n_tower]
     ,Chain(Conv((1,1),n_filter=>64,pad=0,relu),Conv((1,1),64=>9,pad=0),x->reshape(x,:,size(x,4))))|>gpu
end

function resconv2heads(n_filter,n_tower)

     return resconv2heads(Chain(Conv((3,3),3=>n_filter,pad=1),BatchNorm(n_filter,relu)),[resnet(n_filter) for k in 1:n_tower]
     ,Chain(Conv((1,1),n_filter=>2,pad=0),BatchNorm(2,relu),x->reshape(x,:,size(x,4)),Dense(72,37)),
     Chain(Conv((1,1),n_filter=>2,pad=0),BatchNorm(2,relu),x->reshape(x,:,size(x,4)),Dense(72,1,tanh)))|>gpu
end

abstract type predictor end

struct simple_predictor<:predictor
    reseau
end

(m::simple_predictor)(game::GameEnv)=softmax(m.reseau(decoder(game)|>gpu)|>cpu)

struct conv_predictor<:predictor
    reseau
end

function (m::conv_predictor)(game::GameEnv)
    n=getActionNumber(game)
    π=softmax(m.reseau(decoder(game,true)|>gpu)|>cpu)
    p=zeros(n)
    for k in 1:n
        p[k]=π[ftuple[findIndex(game,k)]]
    end
    return p
end

struct conv2h_predictor<:predictor
    reseau
end

function (m::conv2h_predictor)(game::GameEnv)
    n=getActionNumber(game)
    π,v=m.reseau(decoder(game)|>gpu)|>cpu
    π=softmax(π[:,1])
    p=zeros(n)
    for k in 1:n
        move=getNumber(findIndex(game,k))
        p[k]=π[move]
    end
    return p,v
end
