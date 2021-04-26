


mutable struct resnet
    c1
    c2
end

Flux.@functor resnet

resnet(n_filter)=resnet(Dense(n_filter,div(n_filter,4),relu),Dense(div(n_filter,4),n_filter))

 (m::resnet)(x)=relu.(x .+m.c2(m.c1(x)))



mutable struct network
    base
    res
    policy
    value
end


Flux.@functor network

to_cpu(nn::network)=nn|>cpu
function (m::network)(x)
    b=m.base(x)

    for r in m.res
        b=r(b)
    end

    return (m.policy(b),m.value(b))
end


function ressimple(n_filter,n_tower)

     return network(Dense(98,n_filter,relu),[resnet(n_filter) for k in 1:n_tower]
     ,Dense(n_filter,7),
     Dense(n_filter,1,tanh))|>gpu
end

function (m::network)(x::Vector{GameEnv},θ)
    l=length(x)
   batch=zeros(Float32,(98,l))
   @threads for k in 1:l
        @views decoder(x[k],batch[:,k])
   end
   batch=batch|>gpu
   p,v=m(batch)
   v=v|>cpu
   p=softmax(p)|>cpu
   return p,v
end
