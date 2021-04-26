#####################all network output logits###############



const sizeInput=(6,7,2)


mutable struct resnet #########residual block##############
    c1
    b1
    c2
    b2

end

Flux.@functor resnet

#resnet(n_filter)=resnet(Conv((3,3),n_filter=>n_filter,pad=1),BatchNorm(n_filter,relu),Conv((3,3),n_filter=>n_filter,pad=1),BatchNorm(n_filter,relu))

 #(m::resnet)(x)=Chain(m.c1,m.b1,m.c2,m.b2)(x)+x

 resnet(n_filter)=resnet(Conv((3,3),n_filter=>n_filter,pad=1),BatchNorm(n_filter,relu),Conv((3,3),n_filter=>n_filter,pad=1),BatchNorm(n_filter))

  (m::resnet)(x)=relu.(Chain(m.c1,m.b1,m.c2,m.b2)(x)+x)

  mutable struct base #########residual block##############
      c1
      b1
      c2
      b2

  end

  Flux.@functor base

  #resnet(n_filter)=resnet(Conv((3,3),n_filter=>n_filter,pad=1),BatchNorm(n_filter,relu),Conv((3,3),n_filter=>n_filter,pad=1),BatchNorm(n_filter,relu))

   #(m::resnet)(x)=Chain(m.c1,m.b1,m.c2,m.b2)(x)+x

   base(n_filter)=resnet(Conv((1,1),n_filter=>n_filter,pad=0),BatchNorm(n_filter),Conv((5,5),n_filter=>n_filter,pad=2),BatchNorm(n_filter))

    (m::base)(x)=relu.(Chain(m.c1,m.b1)(x)+Chain(m.c2,m.b2)(x))

  # resnet(n_filter)=resnet(Conv((1,1),n_filter=>n_filter,pad=0),BatchNorm(n_filter,relu),Conv((5,5),n_filter=>n_filter,pad=2),BatchNorm(n_filter,relu),Conv((7,7),n_filter=>n_filter,pad=3),BatchNorm(n_filter,relu))
  #
  #  (m::resnet)(x)=relu.(Chain(m.c1,m.b1)(x)+Chain(m.c2,m.b2)(x)+Chain(m.c3,m.b3)(x))+x
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

     return resconv(Chain(Conv((3,3),12=>n_filter,pad=1),base(n_filter)),[resnet(n_filter) for k in 1:n_tower]
     ,Chain(Conv((3,3),n_filter=>64,pad=0,stride=3,relu),Conv((1,1),64=>9,pad=0),x->reshape(x,:,size(x,4))))|>gpu
end

function MPool(x)
  # Input size
  x_size = size(x)
  # Kernel size
  k = x_size[1:end-2]
  # Pooling dimensions
  pdims = PoolDims(x, k)

  return maxpool(x, pdims)
end

function mPool(x)
  # Input size
  x_size = size(x)
  # Kernel size
  k = x_size[1:end-2]
  # Pooling dimensions
  pdims = PoolDims(x, k)

  return meanpool(x, pdims)
end

function resconv2heads(n_filter,n_tower)

     return resconv2heads(Chain(Conv((3,3),2=>n_filter,stride=1,pad=1),BatchNorm(n_filter,relu)),[resnet(n_filter) for k in 1:n_tower]
     ,Chain(Conv((1,1),n_filter=>2,pad=0,stride=1),BatchNorm(2,relu),x->reshape(x,:,size(x,4)),Dense(84,256,relu),Dense(256,7)),
     Chain(Conv((1,1),n_filter=>2,pad=0,stride=1),BatchNorm(2,relu), x->reshape(x,:,size(x,4)),Dense(84,1,tanh)))|>gpu
end


#####     x->cat(GlobalMaxPool()(x),GlobalMeanPool()(x),dims=3),
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
    π=softmax(m.reseau(decoder(game)|>gpu)|>cpu)
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
        p[k]=π[getAction(findIndex(game,k))]
    end
    return p,v
end
