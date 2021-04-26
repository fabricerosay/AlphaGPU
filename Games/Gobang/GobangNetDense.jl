const sizeInput=(2*NN,)


mutable struct resnet #########residual block##############
    c1
    c2

end

Flux.@functor resnet

resnet(n_filter)=resnet(Dense(n_filter,n_filter,relu),Dense(n_filter,n_filter))

 (m::resnet)(x)=relu.(Chain(m.c1,m.c2)(x)+x)


mutable struct resconv2heads ##################residual policy and value###############
    base
    res
    policy
    value
end


Flux.@functor resconv2heads


function (m::resconv2heads)(x)
    b=m.base(x)
    #b=m.res(b)
    for r in m.res
        b=r(b)
    end
    #b=reshape(b,:,size(b,4))
    return (m.policy(b),m.value(b))
end


function resconv2heads(n_filter,n_tower)

     return resconv2heads(Dense(2*NN,n_filter,relu),[resnet(n_filter) for k in 1:n_tower]
     ,Chain(Dense(n_filter,512,relu),Dense(512,NN)),
     Chain(Dense(n_filter,512,relu),Dense(512,256,relu),Dense(256,1,tanh)))|>gpu
end
# function resconv2heads(n_filter,n_tower)
#
#      return resconv2heads(Dense(261,n_filter,relu),Chain(Dense(n_filter,n_filter,relu),
#      Dense(n_filter,n_filter,relu),Dense(n_filter,n_filter,relu),
#      Dense(n_filter,n_filter,relu),Dense(n_filter,n_filter,relu),
#      Dense(n_filter,n_filter,relu),Dense(n_filter,n_filter,relu))
#      ,Chain(Dense(n_filter,512,relu),Dense(512,NN)),
#      Chain(Dense(n_filter,512,relu),Dense(512,1,tanh)))|>gpu
# end

#####     x->cat(GlobalMaxPool()(x),GlobalMeanPool()(x),dims=3),
abstract type predictor end


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
    return p/sum(p),v
end
