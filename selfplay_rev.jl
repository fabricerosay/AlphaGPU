

mutable struct Buffer
    data::Vector{T} where T
    initial_size::Int
    final_size::Int
    current_size::Int
    steps::Int
end

new_buffer(isize,fsize,steps)=Buffer(Vector(),isize,fsize,isize,steps)

function push_buffer(buffer::Buffer,datas,)
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
    startnet,
    samplesNumber = 32000,
    rollout = 128,
    iteration = 100,
    batchsize = 2*4096,
    lr = 0.001,
    epoch = 1,
    )
    
    buffer=new_buffer(2000000,2000000,1)
   
    net=deepcopy(startnet)
    #
    trainingnet = deepcopy(startnet)





    for i = 1:iteration
        println("iteration: $i")

        test_position=mcts_gpu.mcts(net,rollout,samplesNumber)
        push_buffer(buffer,test_position,false)
        println("sample acquis: ",length(test_position))
        println("longueur moyenne des parties: ",length(test_position)/(samplesNumber))
        println("taille du buffer: ",length(buffer.data))






        traininPipe(batchsize,
        trainingnet,
        buffer,
        epoch = epoch,
        lr = lr)



        duel= mcts_gpu.duelnetwork(trainingnet,net,64,1024)
        CUDA.reclaim()

        print("résultat du duel: ", 100 .*duel ./sum(duel))

        if duel[1] > duel[3]
            net = deepcopy(trainingnet)
            reseau = to_cpu(trainingnet)
            if Sys.free_memory() / 2^20 < 700
                println("memory reclaim")
                GC.gc()

            end
            JLD2.@save pwd() * "/Data/reseau$i.json" reseau
        end
    end


end
