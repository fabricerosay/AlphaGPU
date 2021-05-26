

mutable struct Buffer
    data::Vector{T} where T
    initial_size::Int
    final_size::Int
    current_size::Int
    steps::Int
end

new_buffer(isize,fsize,steps)=Buffer(Vector(),isize,fsize,isize,steps)

function push_buffer(buffer::Buffer,datas)
    delta=div((buffer.final_size-buffer.initial_size),buffer.steps)
    buffer.current_size=min(buffer.final_size,buffer.current_size+delta)
    for d in datas
        push!(buffer.data,d)
        if length(buffer.data)>buffer.current_size
            popfirst!(buffer.data)
        end
    end
end

function get_datas(buffer::Buffer)
    return buffer.data
end




function trainingPipeline(
    startnet;
    game="",
    cpuct=2.0,
    noise=0.1f0,
    samplesNumber = 32000,
    rollout = 64,
    iteration = 100,
    batchsize = 2*4096,
    lr = 0.001,
    epoch = 1,
    sizein=98,
    sizeout=7
    )

    buffer=new_buffer(2000000,2000000,1)

    net=deepcopy(startnet)
    #
    trainingnet = deepcopy(startnet)

    #entries=load_pos()

    elocurve=[0.0]

    currentelo=0

    for i = 1:iteration
        println("iteration: $i")
        θ=i/iteration
        ret=mcts_gpu.mcts(net,rollout,samplesNumber,θ=θ,cpuct=cpuct,noise=noise)
        if !ret.valid
            return ret.data
        end
        test_position=ret.data
        push_buffer(buffer,test_position)
        #buffer=test_position
        println("sample acquis: ",length(test_position))
        println("longueur moyenne des parties: ",length(test_position)/(samplesNumber))
        println("taille du buffer: ",length(buffer.data))






        traininPipe(batchsize,
        trainingnet,
        buffer,
        epoch = epoch,
        lr = lr,
        in=sizein,out=sizeout)



        duel= mcts_gpu.duelnetwork(trainingnet,net,64,1024)
        CUDA.reclaim()

        print("résultat du duel: ", 100 .*duel ./sum(duel))
        EA=1024/(duel[1]+0.5*duel[2])
        newelo=currentelo-400*log10(EA-1)
        push!(elocurve,newelo)
        index=(i-1)%1000+1
        if index%10==0
            display(plot(x=0:i,y=elocurve,Geom.point, Geom.line))
            JLD2.@save pwd() * "/Data" *game *"/elocurve$index.json" elocurve
        end
        if duel[1] > duel[3]
            currentelo=newelo
            net = deepcopy(trainingnet)
            reseau = to_cpu(net)
            if Sys.free_memory() / 2^20 < 700
                println("memory reclaim")
                GC.gc()

            end


            JLD2.@save pwd() * "/Data" *game *"/reseau$index.json" reseau
        end

    end


end
