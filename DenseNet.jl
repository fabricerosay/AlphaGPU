#(a::Dense)(x) = a.σ(a.W * x .+ a.b)
mutable struct resnetbatch
    c1
    b1
    c2
    b2
end

Flux.@functor resnetbatch

#resnet(n_filter)=resnet(Dense(n_filter,div(n_filter,4),relu,bias=Flux.Zeros()),Dense(div(n_filter,4),n_filter,bias=Flux.Zeros()))
resnetbatch(n_filter)=resnetbatch(Dense(n_filter,n_filter,bias=Flux.Zeros()),BatchNorm(n_filter,relu),Dense(n_filter,n_filter,bias=Flux.Zeros()),BatchNorm(n_filter,relu))
 (m::resnetbatch)(x)=x .+m.b2(m.c2(m.b1(m.c1(x))))

mutable struct resnet
    c1
    c2
end

Flux.@functor resnet

resnet(n_filter)=resnet(Dense(n_filter,div(n_filter,4),relu,bias=Flux.Zeros()),Dense(div(n_filter,4),n_filter,bias=Flux.Zeros()))
resnetb(n_filter)=resnet(Dense(n_filter,n_filter,bias=Flux.Zeros()),Dense(n_filter,n_filter,bias=Flux.Zeros()))
 (m::resnet)(x)=relu.(x .+m.c2(m.c1(x)))
resnetd(n_filter)=resnet(Dense(n_filter,n_filter,relu,bias=Flux.Zeros()),Dropout(0.25))

 mutable struct resnets
     c1
 end

 Flux.@functor resnets

 resnets(n_filter,nfilter)=resnets(Dense(n_filter,nfilter,relu,bias = Flux.Zeros()))
resnetc(nfilter)=resnets(Conv((3,),nfilter=>nfilter,relu,bias=Flux.Zeros(),
pad=SamePad()))

function (m::resnets)(x,training=false)
    if training
        return relu.(x .+m.c1(x))
    else
        x.=relu.(x .+m.c1(x))
    end
end

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

function ressimpleb(n_filter,n_tower)

     return network(Dense(98,n_filter,relu),[resnetb(n_filter) for k in 1:n_tower]
     ,Dense(n_filter,7),
     Dense(n_filter,1,tanh))|>gpu
end

function ressimplec(in,out,n_filter,n_tower)

     return network(Conv((3,),7=>n_filter,relu,bias=Flux.Zeros(),pad=SamePad()),[resnetc(n_filter) for k in 1:n_tower]
     ,Chain(flatten,Dense(7*n_filter,out)),
     Chain(flatten,Dense(7*n_filter,1,tanh)))|>gpu
end

function ressimples(in,out,n_filter,n_tower)

     return network(Dense(in,n_filter,relu,bias = Flux.Zeros()),[resnets(n_filter,n_filter) for k in 1:n_tower]
     ,Dense(n_filter,out),
     Dense(n_filter,1,tanh))|>gpu
end
function ressimplesb(in,out,n_filter,n_tower)

     return network(Dense(in,n_filter,relu,bias = Flux.Zeros()),[resnet(n_filter) for k in 1:n_tower]
     ,Dense(n_filter,out),
     Dense(n_filter,1,tanh))|>gpu
end
function ressimplessoft(in,out,n_filter,n_tower)

     return network(Dense(in,n_filter,relu,bias = Flux.Zeros()),[resnetb(n_filter,n_filter) for k in 1:n_tower]
     ,Dense(n_filter,out),
     Dense(n_filter,3))|>gpu
end

function ressimpless(n_filter,n_tower)

     return network(Dense(98,n_filter,relu,bias = Flux.Zeros()),[Dense(n_filter,n_filter,relu,bias=Flux.Zeros()) for k in 1:n_tower]
     ,Dense(n_filter,7),
     Dense(n_filter,1,tanh))|>gpu
end

function ressimplebatch(in,n_filter,n_tower)

     return network(Dense(98,n_filter,relu,bias = Flux.Zeros()),[resnetbatch(n_filter) for k in 1:n_tower]
     ,Dense(n_filter,7),
     Dense(n_filter,1,tanh))|>gpu
end

function ressimpled(n_filter,n_tower)

     return network(Dense(98,n_filter,relu,bias = Flux.Zeros()),[resnetd(n_filter) for k in 1:n_tower]
     ,Dense(n_filter,7),
     Dense(n_filter,1,tanh))|>gpu
end

mutable struct networkq
    base
    res
    value
end


Flux.@functor networkq

to_cpu(nn::networkq)=nn|>cpu
function (m::networkq)(x)
    b=m.base(x)

    for r in m.res
        b=r(b)
    end

    return m.value(b)
end

function ressimplesq(in,out,n_filter,n_tower)

     return networkq(Dense(in,n_filter,relu,bias = Flux.Zeros()),[resnets(n_filter,n_filter) for k in 1:n_tower]
     ,Dense(n_filter,out,tanh))|>gpu
end

function ressimplestanh(in,out,n_filter,n_tower)

     return network(Dense(in,n_filter,relu,bias = Flux.Zeros()),[resnets(n_filter,n_filter) for k in 1:n_tower]
     ,Dense(n_filter,out,tanh),
     Dense(n_filter,1,tanh))|>gpu
end

struct Encoder
    base
    res
end

# struct Transition
#     transition
# end
#
#  struct Policy
#     policy
# end
#
#  struct Value
#     value
# end
#
# struct Feature
#     feature
# end

Flux.@functor Encoder
# Flux.@functor Transition
# Flux.@functor Policy
# Flux.@functor Value
# Flux.@functor Feature


Encoder(in,nfilter,ntower)=Encoder(Dense(in,nfilter,relu,bias = Flux.Zeros()),[resnets(nfilter,nfilter) for k in 1:ntower])
function (enc::Encoder)(x)
    b=enc.base(x)
    for r in enc.res
        b=r(b)
    end
    return b
end

Transition(in,nfilter,ntower)=Encoder(Dense(in+nfilter,nfilter,relu,bias = Flux.Zeros()),[resnets(nfilter,nfilter) for k in 1:ntower])

# (trans::Transition)(x)=trans.transition(x)
# Policy(nfilter,out)=Policy(Dense(nfilter,out))
# (pol::Policy)(x)=pol.policy(x)
# Value(nfilter::Int)=Value(Dense(nfilter,1,σ))
# (val::Value)(x)=val.value(x)
# Feature(nfeature,nfilter)=Feature(Dense(nfilter,nfeature,tanh))
# (feat::Feature)(x)=feat.feature(x)


struct MuNet
    encoder
    transition
    policy
    value
    feature
end

Flux.@functor MuNet
to_cpu(nn::MuNet)=nn|>cpu

MuNet(in,out,feature,nfilter,ntower1,ntower2)=MuNet(Encoder(in,nfilter,ntower1),Transition(out,nfilter,ntower2),Dense(nfilter,out),Dense(nfilter,1,σ),Dense(nfilter,feature,tanh))

function (m::MuNet)(x::AbstractArray)
    b=m.encoder(x)
    return (m.policy(b),m.value(b))
end

function(m::MuNet)(x::AbstractArray,y::Vector{T}) where T
    v=Vector{T}()
    b=m.encoder(x)
    base=(m.policy(b),m.value(b),m.feature(b))
    for c in y
        b=m.transition(vcat(b,c))
        push!(v,m.policy(b))
    end
    return base,v
end

mutable struct networkf
    base
    res
    policy
    policy1
    policy2
    policy3
    value
    feature
end


Flux.@functor networkf

to_cpu(nn::networkf)=nn|>cpu
function (m::networkf)(x;training=false)
    b=m.base(x)
    if training
    for r in m.res
        b=r(b,training)
    end
    else
        for r in m.res
            r(b)
        end
    end
    if training
        return (m.policy(b),m.policy1(b),m.policy2(b),m.policy3(b),m.value(b),m.feature(b))
    else
        return (m.policy(b),m.value(b))
    end
end



function ressimplesf(in,out,fsize,n_filter,n_tower)

     return networkf(Dense(in,n_filter,relu,bias = Flux.Zeros()),[resnets(n_filter,n_filter) for k in 1:n_tower]
     ,Dense(n_filter,out),Dense(n_filter,out),Dense(n_filter,out),Dense(n_filter,out),
     Dense(n_filter,1,σ),Dense(n_filter,fsize,tanh))|>gpu
end

mutable struct networkff
    base
    res
    policy
    value
    feature1
    feature2
end


Flux.@functor networkff

to_cpu(nn::networkff)=nn|>cpu
function (m::networkff)(x;training=false)
    b=m.base(x)

    for r in m.res
        b=r(b)
    end
    if training
        return (m.policy(b),m.value(b),m.feature1(b),m.feature2(b))
    else
        return (m.policy(b),m.value(b))#,m.feature1(b),m.feature2(b))
    end
end



function ressimplesff(in,out,n_filter,n_tower)

     return networkff(Dense(in,n_filter,relu,bias = Flux.Zeros()),[resnets(n_filter,n_filter) for k in 1:n_tower]
     ,Dense(n_filter,out),
     Dense(n_filter,1,tanh),Dense(n_filter,64,tanh),Dense(n_filter,out))|>gpu
end


mutable struct network_rec
    base
    res
    rec
    policy
    value
end


Flux.@functor network_rec

to_cpu(nn::network)=nn|>cpu
function (m::network_rec)(x,play)
    b=m.base(x)

    for r in m.res
        b=r(b)
    end
    m.rec.state=(b,b)
    return (m.policy(m.rec(play)),m.value(m.play(b)))
end



function ressimples_rec(in,out,n_filter,n_tower)

     return network_rec(Dense(in,n_filter,relu,bias = Flux.Zeros()),[resnets(n_filter,n_filter) for k in 1:n_tower],
     LSTM(7,n_filter),Dense(n_filter,out),
     Dense(n_filter,1,tanh))|>gpu
end

function fast_eval(m::networkf,x)
    b=m.base(x)
    bmem=copy(b)
    for r in m.res
        bamul!(bmem,r.c1.W,b)
        b.=relu.(bmem) .+b
        bmem.=b
    end
    return (m.policy(b),m.value(b))
end


mutable struct snetwork2{A,B,C,D,E,F}
    base::A
    res::Vector{B}
    policy::C
    policy_bias::D
    value::E
    value_bias::F
end

snetwork2(n::Int,k::Int)=snetwork2(CuArray(randn(Float32,(n,84))/512),[CuArray(randn(Float32,(n,n))/512) for j in 1:k],CuArray(randn(Float32,(7,n))/512),
CuArray(randn(Float32,7)/512),CuArray(randn(Float32,(1,n))/512),CuArray(randn(Float32,1)/512))

snetwork2(n::Int,k::Int,cpu=false)=cpu==false ? snetwork2(n::Int,k::Int) : snetwork2(randn(Float32,(n,189))/512,[randn(Float32,(n,n))/512 for j in 1:k],randn(Float32,(81,n))/512,
randn(Float32,81)/512,randn(Float32,(1,n))/512,randn(Float32,1)/512)

function (m::snetwork2)(x::T;training=false) where T<:CuArray
    b=relu.(m.base*x)

    for w in m.res
        b.=relu.(b.+relu.(w*b))
    end
    #return m.policy*b,tanh.(m.value*b)
    policy,value=m.policy*b.+m.policy_bias,σ.(m.value*b.+m.value_bias)
    #value=sum(softmax!(value).*(0f0,0.5f0,1f0),dims=1)
    return policy,value
end

function (m::snetwork2)(x::T;training=false) where T<:Array
    b=relu.(m.base*x)

    for w in m.res
        b.=relu.(b.+relu.(w*b))
    end
    #return m.policy*b,tanh.(m.value*b)
    policy,value=softmax(m.policy*b.+m.policy_bias),σ.(m.value*b.+m.value_bias)
    #value=sum(softmax!(value).*(0f0,0.5f0,1f0),dims=1)
    return policy,value
end

function (m::snetwork2)(x::T,prior,value;training=false) where T<:CuArray
    b=relu.(m.base*x)
    c=similar(b)
    for w in m.res
        CUBLAS.gemm!('N','N',1f0,w,b,1f0,c)
        b.=relu.(b.+relu.(c))
    end
    #return m.policy*b,tanh.(m.value*b)
    #return CUBLAS.gemm!('N','N',1f0,m.policy,b,1f0,prior).+m.policy_bias,tanh.(CUBLAS.gemm!('N','N',1f0,m.value,b,1f0,value).+m.value_bias)

    #return m.policy*b.+m.policy_bias,tanh.(m.value*b.+m.value_bias)
end

function convert_back(net::networkf)
    return snetwork2(net.base.weight,[w.c1.weight for w in net.res],net.policy.weight,net.policy.bias,net.value.weight,net.value.bias)
end

function convert_back(net::snetwork2)
    return snetwork2(net.base.weight,[w.c1.weight for w in net.res],net.policy.weight,net.policy.bias,net.value.weight,net.value.bias)
end

function convert_back_cpu(net::networkf)
    return snetwork2(Array(net.base.weight),[Array(w.c1.weight) for w in net.res],Array(net.policy.weight),Array(net.policy.bias),Array(net.value.weight),Array(net.value.bias))
end


module unroll
export @unroll
copy_and_substitute_tree(e, varname, newtext) = e

function copy_and_substitute_tree(e::Symbol, varname, newtext)
    if e == varname

        return newtext
    # elseif e==:x
    #     println("x")
    #     return Meta.parse("x$newtext")
    else
        return e
    end
end

#copy_and_substitute_tree(e, varname, newtext) =
    #    e == :x ? Meta.parse("x$newtext") : e
function copy_and_substitute_tree(e::Expr, varname, newtext)
    e2 = Expr(e.head)
    for subexp in e.args
        push!(e2.args, copy_and_substitute_tree(subexp, varname, newtext))
    end
    e2
end


macro unroll(expr,var...)
    if expr.head != :for || length(expr.args) != 2 ||
        expr.args[1].head != :(=) ||
        typeof(expr.args[1].args[1]) != Symbol ||
        expr.args[2].head != :block
        error("Expression following unroll macro must be a for-loop as described in the documentation")
    end
    varname = expr.args[1].args[1]
    ret = Expr(:block)
    for k in Core.eval(@__MODULE__, expr.args[1].args[2])
        e2=expr.args[2]
        for v in var
            e2=copy_and_substitute_tree(e2, v,Meta.parse(string(v)*"$k"))
        end
        e2 = copy_and_substitute_tree(e2,varname,k)
        push!(ret.args, e2)
    end
    esc(ret)
end

macro prior(N)
    ret = Expr(:block)
    for k in 1:N
        e=Expr(:(=),Symbol(:p,k),:(vnodesStats.prior[i,nindex,$k]))
        push!(ret.args, e)
    end
    esc(ret)
end
macro q(N)
    ret = Expr(:block)
    for k in 1:N
        e=Expr(:(=),Symbol(:q,k),:(vnodesStats.q[i,nindex,$k]))
        push!(ret.args, e)
    end
    esc(ret)
end

macro heart(N)
    ret = Expr(:block)
    for k in 1:N
        e=Expr(:(=),:top,Expr(:call,:*,:λ,Symbol(:p,k)))
        push!(ret.args, e)
        e=Expr(:(=),:bot,Expr(:call,:-,:α,Symbol(:q,k)))
        push!(ret.args, e)
        e=Expr(:(+=),:g,Expr(:call,:/,:top,:bot))
        push!(ret.args, e)
        e=Expr(:(-=),:S,Expr(:call,:/,:top,Expr(:call,:*,:bot,:bot)))
        push!(ret.args, e)
    end
    esc(ret)
end

end
function test2(vnodesStats,i)
    g=0
    S=0
    p1=1
    p2=2
    q1=0.5
    q2=1
    α=3
    λ=2
    #unroll.@prior(2)
    unroll.@heart(2)

 end
