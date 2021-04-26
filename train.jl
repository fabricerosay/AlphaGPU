function lossvalue(net,x,y)
    p,v=net(x)
    return Flux.mse(v,y[2])
end

function losspolicy(net,x,y)
    p,v=net(x)
    return Flux.logitcrossentropy(p,y[1])
end

function lossmini(net,x,y)
    p,v=net(x)
    return Flux.mse(w,y[3])
end




function making_batch(p,batchsize;value=true)
    r=[]
    I=CartesianIndices(sizeInput)
    for a in partition(p,batchsize)

        tmpx=zeros(sizeInput...,length(a))

        tmpy=zeros(maxActions,length(a))
        tmpw=zeros(9,length(a))
        if value
            tmpz=zeros(1,length(a))
        end

        for (k,(x,y)) in enumerate(a)

            tmpx[I,k].=x[I,1]
            if value
                tmpy[:,k].=y[1]
                tmpz[:,k].=y[2]
                #tmpw[:,k].=y[3]
            else
                tmpy[:,k].=y
            end
        end
        if value
            push!(r,(cu(tmpx),(cu(tmpy),cu(tmpz))))#,cu(tmpw))))
        else
            push!(r,(cu(tmpx),cu(tmpy)))
        end
    end
    GC.gc(true)
    CuArrays.reclaim()
    return r
end


function traininPipe(batchsize,net,p;epoch=1, lr=0.001,value=true,βloss=1) #######we train on all data easy to chnage to random batch#####
    t=min(div(length(p),10),1000)
    r=deepcopy(p[1:end-t]) ######## we keep 1000 sample to test ############
    #r[test=making_batch(p[end-t:end],batchsize,value=value)
    lossTot(net,x,y)=losspolicy(net,x,y)+βloss*lossvalue(net,x,y)+0.0001f0*sum(x->sum(abs2,x),Flux.params(net))#+-0.1f0*Flux.logitcrossentropy(net(x),softmax(net(x)))

    l=0
    cpt=0
    #evalcb() = @show(accuracy(p[end-t:end],net,value),sum(lossvalue(net,cu(b[1]),b[2]) for b in p[end-1000:end])/1000)
    evalcb() = @show(sum(lossTot(net,cu(b[1]),cu(b[2])) for b in p[1:1000])/1000,sum(lossvalue(net,cu(b[1]),cu(b[2])) for b in p[end-1000:end])/1000,sum(losspolicy(net,cu(b[1]),cu(b[2])) for b in p[end-1000:end])/1000)
    nbbatches=Int(round(length(r)*1.5))
    for i  in 1:epoch
         #####callback to monitor training

        #for j in 1:nbbatches



        shuffle!(r)

        dat=making_batch(r,batchsize,value=value)
        # if j%div(nbbatches,5)==0
        println("Accuracy on the trained data: ",accuracy(p[1:t],net,value))
        #     println("accuracy: ", accuracy(p[end-t:end],net,value))
        #     println("loss value: ", mean(lossvalue(net,b...) for b in test))
        # end
        #evalcb() = @show(accuracy(p[end-t:end],net,value),mean(lossvalue(net,b...) for b in test))
        Flux.train!((x,y)->lossTot(net,x,y),Flux.params(net),dat,ADAM(lr),cb=throttle(evalcb,60))

        l+=mean(lossTot(net,(d|>gpu)...) for d in dat)

        cpt+=1
        #end
        println("average loss ",l/cpt)

    end
    GC.gc(true)
    CuArrays.reclaim()
    println("loss moyen ",l/(epoch))
end


function accuracy(p,net,value)
    justes=0
    ind=0
    for pos in p
        x,y=pos
        if value
            justes+=argmax(((net(x|>gpu)[1])|>cpu)[:,1]).==argmax(y[1])
        else
            justes+=argmax(((net(x|>gpu))|>cpu)[:,1]).==argmax(y)
        end

        ind+=1
    end
    justes/ind
end

function progressiveTraining(actor,lr=0.0001)
    r=recoverData(30,34,"UTTT")
    traininPipe(1024,actor,r,epoch=1,lr=lr,value=true,βloss=1)
    r=recoverData(25,29,"UTTT")
    traininPipe(1024,actor,r,epoch=1,lr=lr,value=true,βloss=1)
    r=recoverData(20,24,"UTTT")
    traininPipe(1024,actor,r,epoch=1,lr=lr,value=true,βloss=1)
    r=recoverData(15,19,"UTTT")
    traininPipe(1024,actor,r,epoch=1,lr=lr,value=true,βloss=1)


end
