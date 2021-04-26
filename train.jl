
function lossvalue(net,x,y)
    p,v=net(x)
    return Flux.mse(v,y[2])
end

function losspolicy(net,x,y)
    p,v=net(x)
    return Flux.logitcrossentropy(p,y[1])
end

function lossTot(net,x,y)
    p,v=net(x)
    return Flux.logitcrossentropy(p,y[1])+Flux.mse(v,y[2])#+0.0001f0*sum(x->sum(abs2,x),Flux.params(net))
end

function making_batch(q,batchsize;value=true)
    r=[]

    I=CartesianIndices(sizeInput)
    for a in partition(q,batchsize)

        tmpx=zeros(Float32,(98,length(a)))
        tmpy=zeros(Float32,(7,length(a)))
        tmpr=zeros(Float32,(1,length(a)))


        for (k,(x,y)) in enumerate(a)

                tmpx[:,k].=x

                tmpy[:,k].=y[1]
                tmpr[:,k].=y[2]


        end
        push!(r,(tmpx|>gpu,(tmpy|>gpu,tmpr|>gpu)))
    end
    GC.gc(true)
    return r
end


function traininPipe(batchsize,net,p;epoch=1, lr=0.001,value=true,βloss=0.0001f0,generation=1)
     tuation(net,x,y)+0.0001f0*sum(x->sum(abs2,x),Flux.params(net))
    l=0
    cpt=0

    opt=Flux.Optimise.Optimiser(ADAM(lr),WeightDecay(0.0001))

    for i  in 1:epoch

         println("epoque: ",i)

         q=sample(get_datas(p),min(2000000,length(p.data)))

          nbsamples=length(q)
          dat=making_batch(q,batchsize)
         println("batch number: ",div(nbsamples,batchsize))

        Flux.train!((x,y)->lossTot(net,x,y),Flux.params(net),dat,opt)

        GC.gc(true)
        CUDA.reclaim()
        @show(sum(lossTot(net,b[1],b[2]) for b in dat)/length(dat))
    end
    GC.gc(true)
    testmode!(net)

end
