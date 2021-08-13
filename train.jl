
function lossvalue(net,x,y)
    p,v=net(x)
    return Flux.mse(v,y[2])
end

function losspolicy(net,x,y)
    p,v=net(x)
    return Flux.logitcrossentropy(p,y[1])
end

function lossTot(m,x,y,K)
    b=m.encoder(x[1],true)
    p,v,f=m.policy(b),m.value(b),m.feature(b)
    loss=Flux.logitcrossentropy(p,y[1])+Flux.mse(v,y[2])+0.001f0*Flux.mse(f,y[3])
    for k in 1:K
        b=vcat(b,x[2][k])
        b=m.transition(b,true)
        loss+=Flux.logitcrossentropy(m.policy(b),y[4][k])/Float32(K)
    end
    #loss+=0.1f0*Flux.logitcrossentropy(p,y[4][1])
    # loss+=0.1f0*Flux.logitcrossentropy(p2,y[4][2])
    # loss+=0.1f0*Flux.logitcrossentropy(p3,y[4][3])
    return loss
end




function traininPipe(batchsize,net,p;in=98,out=7,fsize=1,epoch=1, lr=0.001,value=true,βloss=0.0001f0,generation=1,actor2=nothing)


    opt=Flux.Optimise.Optimiser(ADAM(lr),WeightDecay(0.0001))
    #q=sample(p,4*length(p))

    K=p.unrollsteps
    for i  in 1:epoch

         println("epoque: ",i)
         t=time()
         L=length_buffer(p)
        q=sample(p.pool[1:L],min(2000000,L))
         #dat=making_batch(q,batchsize,in=in,out=out)
         nbsamples=length(q)
         t=time()-t
         L=div(nbsamples,batchsize)
         println("batch number: ",div(nbsamples,batchsize))
         totloss=0
         println("batching time $t")
         println("good training param")
         t=time()
         tmpx=zeros(Float32,(in,batchsize))
         tmpy=zeros(Float32,(out,batchsize))
         tmpyy=[zeros(Float32,(out,batchsize)) for k in 1:K]
         tmpm=[zeros(Float32,(out,batchsize)) for k in 1:K]
         tmpf=zeros(Float32,(fsize,batchsize))
         tmpr=zeros(Float32,(1,batchsize))
         tmpx_g=CuArray(zeros(Float32,(in,batchsize)))
         tmpy_g=CuArray(zeros(Float32,(out,batchsize)))
         tmpyy_g=[CuArray(zeros(Float32,(out,batchsize))) for k in 1:K]
         tmpm_g=[CuArray(zeros(Float32,(out,batchsize))) for k in 1:K]
         tmpf_g=CuArray(zeros(Float32,(fsize,batchsize)))
         tmpr_g=CuArray(zeros(Float32,(1,batchsize)))
         for (cpt,a) in enumerate(partition(q,batchsize))
             if cpt>=L
                 break
            end


             for (k,sp) in enumerate(a)

                     tmpx[:,k].=sp.state
                     tmpy[:,k].=sp.policy[:,1]
                     tmpr[:,k].=sp.value
                     tmpf[:,k].=sp.fstate
                     for j in 1:K
                         tmpyy[j][:,k].=sp.policy[:,j+1]
                         c=sp.move[j]
                         if c!=0
                             tmpm[j][sp.move[j],k]=1
                        end
                     end
             end

             copyto!(tmpx_g,tmpx)
             copyto!(tmpy_g,tmpy)
             copyto!(tmpr_g,tmpr)
             copyto!(tmpf_g,tmpf)
             for j in 1:K
                 copyto!(tmpyy_g[j],tmpyy[j])
                 copyto!(tmpm_g[j],tmpm[j])
             end

             totloss+=custom_train!((x,y)->lossTot(net,x,y,K),Flux.params(net),[((tmpx_g,tmpm_g),(tmpy_g,tmpr_g,tmpf_g,tmpyy_g))],opt)#,tmpf_g,tmpyy_g))],opt)
             for j in 1:K
                 tmpm[j].=0
             end
         end
         tmpx=nothing
         tmpy=nothing
         #tmpff=zeros(Float32,(out,batchsize))
         tmpf=nothing
         tmpr=nothing
         CUDA.unsafe_free!(tmpx_g)
         CUDA.unsafe_free!(tmpy_g)
         CUDA.unsafe_free!(tmpr_g)
         CUDA.unsafe_free!(tmpf_g)

         GC.gc(true)

        # if actor2!=nothing
        #     totloss2=custom_train!((x,y)->lossTot(actor2,x,y),Flux.params(actor2),dat,opt)
        #     println("total loss2: ",totloss2)
        # end
        #CUDA.unsafe_free!(dat)
        GC.gc(true)
        CUDA.reclaim()
        #@show(sum(lossTot(net,b[1],b[2]) for b in dat)/length(dat))
        t=time()-t
        println("total loss: ",totloss/(L-1))
        println("training time :$t")
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
    #l=length(ps)-4
    # for k in Iterators.reverse(2:l-4)
    #     w=ps[k]
    #     length(size(w))==1 && continue
    #     importance=sum(abs,w,dims=2)
    #     importance./=0.5f0*maximum(importance)
    #     unit=similar(importance)
    #     unit.=1
    #     importance.=min.(importance,unit)
    #     fullimportance.*=importance
    #
    #    gs[w].*=fullimportance
    # for k in 2:l-4
    #      w=ps[k]
    #      gs[w][freeze].=0
    #      w[freeze].=0
    # end

    Flux.update!(opt, ps, gs)

  end
  return total_loss/length(data)
end
