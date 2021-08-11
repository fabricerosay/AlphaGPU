function trainingPipeline(
    net,trainingnet,buffer,generation,currentelo=-1000;
    game="",
    cpuct=2.0,
    noise=0.1f0,
    samplesNumber = 32000,
    rollout = 64,
    iteration = 100,
    batchsize = 4096,
    lr = 0.001,
    epoch = 1,
    sizein=98,
    sizeout=7,
    fsize=1
    )



    #net=deepcopy(startnet)
    #
    #trainingnet = deepcopy(startnet)

    #entries=load_pos()

    elocurve=[]

    passing=false
    oldacc=100

    for i = generation:generation
        println("iteration: $i")
        θ=1-i/iteration
        #Init=mcts_gpu.init(samplesNumber,rollout)
        mcts_gpu.mcts(net,rollout,samplesNumber,buffer,cpuct=cpuct,noise=noise)
        println("fin de la première volée")

        #println("sample acquis: ",length(test_position))
        #println("longueur moyenne des parties: ",length(test_position)/(samplesNumber))
        println("taille du buffer: ",length_buffer(buffer))

        #return buffer




        traininPipe(batchsize,
        trainingnet,
        buffer,
        epoch = epoch,
        lr = lr,
        in=sizein,out=sizeout,fsize=fsize,actor2=nothing)

        index=(i-1)%1000+1
        if true

        duel= mcts_gpu.duelnetwork(trainingnet,net,32,1024,-1)
        #acc=full_evaluation(net,entries,600)



        GC.gc(true)
        print("résultat du duel: ", 100 .*duel ./sum(duel))
        #println("accuracy: $acc")
        EA=1024/(duel[1]+0.5*duel[2])
        newelo=-400*log10(EA-1)+currentelo
        push!(elocurve,newelo)

        if false#index%2==0
            display(plot(x=1:i,y=elocurve,Geom.point, Geom.line))
            JLD2.@save pwd() * "/Data" *game *"/elocurve$index.json" elocurve
        end
        if newelo>currentelo#duel[1] > duel[3]
            currentelo=newelo
            passing=true
            net = deepcopy(trainingnet)

        end
            # if newelo>400
            #     currentelo=0
            #     testnet=deepcopy(trainingnet)
            # end
        #end
    end
        if true

            reseau = to_cpu(trainingnet)
            # if testnet!=nothing
            #     reseau2 = to_cpu(testnet)
            #     JLD2.@save pwd() * "/Data" *game *"/reseau_big$index.json" reseau2
            # end
            if Sys.free_memory() / 2^20 < 700
                println("memory reclaim")
                GC.gc()

            end


            JLD2.@save pwd() * "/Data" *game *"/reseau$index.json" reseau

        end

        # resave=net|>cpu
        # CUDA.device_reset!(dev)
        # trainingnet=reseau|>gpu
        # net=resave|>gpu
    end

    return net,trainingnet,passing,currentelo
end
