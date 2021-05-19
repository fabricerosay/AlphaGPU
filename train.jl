
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

function making_batch(q,batchsize;value=true,in,out)
    r=[]


    for a in partition(q,batchsize)

        tmpx=zeros(Float32,(in,length(a)))
        tmpy=zeros(Float32,(out,length(a)))
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


function traininPipe(batchsize,net,p;in=98,out=7,epoch=1, lr=0.001,value=true,βloss=0.0001f0,generation=1)


    opt=Flux.Optimise.Optimiser(ADAM(lr),WeightDecay(0.0001))
    #q=sample(p,4*length(p))
    testmode!(net,false)
    for i  in 1:epoch

         println("epoque: ",i)

         q=sample(get_datas(p),min(2000000,length(p.data)))
         dat=making_batch(q,batchsize,in=in,out=out)
          nbsamples=length(q)

         println("batch number: ",div(nbsamples,batchsize))

        totloss=custom_train!((x,y)->lossTot(net,x,y),Flux.params(net),dat,opt)

        GC.gc(true)
        CUDA.reclaim()
        #@show(sum(lossTot(net,b[1],b[2]) for b in dat)/length(dat))
        println("total loss: ",totloss)
    end
    GC.gc(true)
    testmode!(net,true)

end


function custom_train!(loss, ps, data, opt)
  # training_loss is declared local so it will be available for logging outside the gradient calculation.
  local training_loss
  total_loss=0
  for d in data
    gs = gradient(ps) do
      training_loss = loss(d...)
      return training_loss
    end
    total_loss+=training_loss
    # fullimportance=CuArray(ones(Float32,(size(ps[1])[1],1)))
    # fullimportance.=1
    # l=length(ps)-4
    # for k in Iterators.reverse(1:l-4)
    #     w=ps[k]
    #     length(size(w))==1 && continue
    #     importance=sum(abs,w,dims=2)
    #     importance./=0.5f0*maximum(importance)
    #     unit=similar(importance)
    #     unit.=1
    #     importance.=min.(importance,unit)
    #     fullimportance.*=importance
    #     gs[w].*=fullimportance
    # end

    Flux.update!(opt, ps, gs)

  end
  return total_loss/length(data)
end
