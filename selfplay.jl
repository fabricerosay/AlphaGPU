@inline function pitNetwork(
    p1,
    p2,
    n,
    temptresh,
    affichage = 50;
    usepuct = false,
    vanille = (false, 100),
)

    v = 0
    nul = 0
    d = 0
    reward = 0
    f = 0
    testmode!(p1, true)
    testmode!(p2, true)
    net2 = conv2h_predictor(p2)

    if usepuct
        puct2 = TreePolicy("puct", true, 0, 0.3, 0, 1, 0, p2,4)
        #puct1=TreePolicy("puct",true,0,0.3,0,1,0,p1)
    end
    if vanille[1]
        net1 = x->vanilla(x,vanille[2])
    elseif usepuct
        net1 = TreePolicy("puct",true,0,0.3,0,1,0,p1,4)
    else
        net1 = conv2h_predictor(p2)
    end
    for i = 1:n
        current = GameEnv((-1)^i)

        if i % affichage == 0
            println("game $i", (v, nul, d))
        end
        k = 0
        #lp = legalPlays(current)
        while !winner(current)[1]
            if current.player == 1

                if vanille[1]
                    q = net1(current)[1]
                elseif usepuct
                    q = net1(current,100)[1]
                else
                    q=net1(current)[1]
                end

                if k < temptresh
                    if vanille[1] || usepuct
                        moveIndex = getAction(sample(1:maxActions, Weights(q)))
                     else
                        moveIndex = moveIndex = findIndex(
                            current,
                            sample(1:getActionNumber(current), Weights(q)),
                        )
                    end

                else
                    if vanille[1] || usepuct
                        moveIndex = getAction(argmax(q))
                     else
                    #
                        moveIndex = findIndex(current, argmax(q))
                     end
                end
                k += 1
            else

                if !usepuct
                    q = net2(current)[1]

                else
                    q = puct2(current, 100)[1]
                end
                if k < temptresh
                    if !usepuct
                        moveIndex = findIndex(
                            current,
                            sample(1:getActionNumber(current), Weights(q)),
                        )
                    else
                        moveIndex = getAction(sample(1:maxActions, Weights(q)))
                    end
                else
                    if !usepuct
                        moveIndex = findIndex(current, argmax(q))
                    else
                        moveIndex = getAction(argmax(q))
                    end
                end

                k += 1
            end

            current = playIndex(current, moveIndex)
        end
        finished = winner(current)[2]
        if finished == -1
            v += 1

        elseif finished == 0
            nul += 1

        else
            d += 1

        end
    end
    testmode!(p1, :auto)
    testmode!(p2, :auto)
    return v, nul, d

end

#############play games till having n samples of data, workers is the number of games in parallel (on the gpu) #####
############# could also multithread but sometimes it gets instable and doesn't add that much speed ########

function batchedSelfplay(
    rollout,
    n,
    workers,
    net,
    temptresh,
    λ,
    cpuct,
    value = true,
    root = nothing,
)
    r = []
    resglobal = [0, 0, 0]
    rtemp = [[] for k = 1:workers]
    rtempbis = [[] for k = 1:workers]
    tempstate = zeros(sizeInput..., workers)######## need a hack to automate this
    i = 1


    current =[GameEnv() for k in 1:workers]# [playIndex(GameEnv(),getAction(sample(1:NN,Weights(root[1])))) for k = 1:workers]

    #current=vcat([GameEnv() for k in 1:div(workers,2)],[playIndex(GameEnv(),rand(1:maxActions)) for k in 1:div(workers,2)])
    k = zeros(workers)
    step = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    #### moins de bruit dirichlet ######
    puct = TreePolicy("puct", true, 0.25, 1, 0, cpuct, 0, net,4)
    puctfast = TreePolicy("puct", true, 0, 1, 0, cpuct, 0, net,16)
    cpt = 0
    logsample = true
    res = [(false, 0) for k = 1:workers]
    tour = 1
    testmode!(net, true)
    while true
        #logsample = rand() < 0.25
        # for j in 1:workers
        #      tempstate[:,j].=decoder(current[j])[:,:,:,1]
        #  end
        # qplay=softmax(net(tempstate|>gpu)[1])|>cpu
        ##### oscillatorycap selfplay à la katago sofar best way was following ExIt ie follow policy not puct
        tour += 1
        if Sys.free_memory() / 2^20 < 700
            println("memory reclaim")
            GC.gc()
            CuArrays.reclaim()
        end
        if logsample
            qv = puct(current, rollout)
        else
            qv=puctfast(current,200)
        end

        for j = 1:workers


            if k[j] < temptresh

                # lp=[getNumber(el) for el in legalPlays(current[j])]
                # moveIndex=sample(lp,Weights(qplay[:,j][lp]))
                moveIndex = getAction(sample(1:maxActions, Weights(qv[j][1])))

            else

                # lp=[getNumber(el) for el in legalPlays(current[j])]
                # moveIndex=lp[argmax(qplay[:,j][lp])]
                moveIndex = getAction(argmax(qv[j][1]))

            end
            k[j] += 1
            if logsample
                push!(
                    rtemp[j],
                    (
                        decoder(current[j]),
                        (qv[j][1], qv[j][2], current[j].player),
                    ),
                )
            end
            ####### Randomize starting positions good ? ##########
            if k[j]<=4
                moveIndex=findIndex(current[j],rand(1:getActionNumber(current[j])))
            end
            current[j] = playIndex(current[j], moveIndex)
            res[j] = winner(current[j])
        end
        j = 1
        while j <= length(current)
            if res[j][1]
                resglobal[res[j][2]+2] += 1

                for pos in rtemp[j]
                    x, (y, z, w) = pos

                    push!(rtempbis[j], (x, (y, 0.5*z+0.5*w*res[j][2])))
                end
                cpt += 1
                if cpt / n * 100 > step[i]
                    println("avancement ", cpt / n * 100, "%")
                    i += 1
                end
                if n - cpt > workers - 1
                    rtemp[j] = []
                    res[j] = (false, 0)
                    k[j] = 0
                    #coup=getAction(sample(1:81,Weights(root[1])))
                    current[j] = GameEnv()# playIndex(GameEnv(),coup)

                    j+=1
                else
                    deleteat!(rtemp, j)
                    deleteat!(res, j)
                    deleteat!(k, j)
                    deleteat!(current, j)
                end
                if cpt >= n
                    r = reduce(vcat, rtempbis)
                    # for k in 1:div(n,81)
                    #         push!(
                    #             r,
                    #             (decoder(GameEnv()), (root[1]/sum(root[1]), root[2]))
                    #         )
                    # end
                    println("résultat global", resglobal)
                    testmode!(net, :auto)
                    return r
                end

            else
                j += 1
            end
        end

        workers = length(current)
    end

    if Sys.free_memory() / 2^20 < 700
        println("memory reclaim")
        GC.gc()
        CuArrays.reclaim()
    end
    return r
end



function trainingPipeline(
    net,
    r = nothing,
    value = true;
    nbworkers = 100,
    game = "UTTT",
    bufferSize = 500000,
    samplesNumber = 1000,
    rollout = 1000,
    cpuct = 2,
    iteration = 100,
    chkfrequency = 1,
    batchsize = 512,
    lr = 0.001,
    epoch = 1,
    temptresh = 12,
)
    if r == nothing
        r = []
        train = false
    else
        train=true
    end
    trainingnet = deepcopy(net)
    vanillarollout = 100
    root = (zeros(NN),0)
    if Sys.free_memory() / 2^20 < 300
        println("memory reclaim")
        GC.gc()
        CuArrays.reclaim()
    end
    t = 4

    for i = 1:iteration
        # nr,nv=vanilla(GameEnv(),100000)
        # root=(root[1].+nr,(((i-1)*root[2]+nv)/i))
        println("iteration: $i")
        # println("état racine")
        # show(stdout, "text/plain",reshape(root[1]/sum(root[1]),(N,N)))
        # println("valeur départ", root[2])
        # show(stdout, "text/plain",reshape(root.actionsN,(9,9)))
        λ = 0#σ(2*(i/iteration-0.7))

        test_position = batchedSelfplay(
            rollout,
            samplesNumber,
            nbworkers,
            net,
            temptresh,
            λ,
            cpuct,
            value,
            root,
        )
        r = vcat(r, test_position)

        if length(r) > bufferSize
            r = r[length(r)-bufferSize:end]
        end
        # if i%2==0
        #     #rollout=min(rollout+50,800)
        #     t=min(t+1,20)
        # end
        if i % 1 == 0
            if Sys.free_memory() / 2^20 < 700
                println("memory reclaim")
                GC.gc()
                CuArrays.reclaim()
            end

            traininPipe(
                batchsize,
                trainingnet,
                r,
                epoch = epoch,
                lr = lr,
                value = value,
                βloss=1
            )
        end

        if i % chkfrequency == 0
            if Sys.free_memory() / 2^20 < 700
                println("memory reclaim")
                GC.gc()
                CuArrays.reclaim()
            end
            p = deepcopy(test_position)
            if Sys.free_memory() / 2^20 < 700
                println("memory reclaim")
                GC.gc()
                CuArrays.reclaim()
            end
            tag = (i-1) % 100+1
            @save pwd() * "/Games/" * game * "/Data/data$tag.json" p
            p = []
        end
        if i %  1== 0
            duel = pitNetwork(net, trainingnet, 40, temptresh,usepuct=true)

                duel2 = pitNetwork(
                    net,
                    trainingnet,
                    40,
                    temptresh,
                    vanille = (true, vanillarollout),
                    usepuct=false
                )
                print("résultat contre mcts $vanillarollout", duel2)
                if duel2[1] > duel2[3]
                    vanillarollout += 100
                end

            print("résultat du duel: ", duel)

            if duel[1] > duel[3]
                net = deepcopy(trainingnet)
            end
        end
            reseau = trainingnet |> cpu
            if i % chkfrequency == 0
                if Sys.free_memory() / 2^20 < 700
                    println("memory reclaim")
                    GC.gc()
                    CuArrays.reclaim()
                end
                #tag = div(i,4)
                @save pwd() * "/Games/" * game * "/Data/reseau$i.json" reseau
            end

        if i >= 1
            bufferSize = min(1000000, Int(round(bufferSize * 1.1)))
        end
    end

end

function recoverData(start, finish, game, plus = "")
    r = []
    for i = start:finish
        @load pwd() * "/Games/" * game * "/Data/data" * plus * "$i.json" p
        r = vcat(r, p)
    end
    return r
end


function bootstrapSelfplay(rollout, n, temptresh)
    r = []
    rtemp = []
    i = 1
    step = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    puct = TreePolicy("classic", true, 0.25, 0.3, 0, 1, 0, nothing)
    cpt = 0
    current = GameEnv()
    res = winner(current)
    k = 0
    logsample=true
    for j = 1:n
        while !res[1]
            if rand()<2
                logsample=true
                qv = puct(current, rollout)
            else
                logsample=false
                qv = puct(current, 100)
            end

            if k < temptresh


                moveIndex = getAction(sample(1:maxActions, Weights(qv[1])))

            else


                moveIndex = getAction(argmax(qv[1]))

            end
            k += 1
            if logsample
                push!(rtemp, (decoder(current), (qv[1], qv[2])))
            end
            current = playIndex(current, moveIndex)
            res = winner(current)
        end


        for pos in rtemp
            x, (y, w) = pos
            push!(r, (x, (y, w)))
        end
        current = GameEnv()
        res = winner(current)
        k = 0
        rtemp = []
    end
    return r
end

function bootstrapSelfplayParallel(rollout, n, temptresh, workers)

    r = [[] for i = 1:workers]

    @threads for i = 1:workers
        r[i] = vcat(r[i], bootstrapSelfplay(rollout, n, temptresh))
    end
    for i = 1:workers
        tag = i
        p = r[i]
        @save pwd() * "/Games/UTTT/Data/datavanilla$tag.json" p
    end

end


function trainingPipelinePBT(
    net,
    r = nothing,
    value = true;
    nbworkers = 100,
    game = "UTTT",
    bufferSize = 500000,
    samplesNumber = 1000,
    rollout = 1000,
    cpuct = 2,
    iteration = 100,
    chkfrequency = 1,
    batchsize = 512,
    lr = 0.001,
    epoch = 1,
    temptresh = 4,
    npopulation=8
)
    multiplie=[0.8,1.2]
    trainingnets = [deepcopy(net) for i in 1:npopulation]
    lr=[rand(multiplie)*0.01 for k in 1:npopulation ]
    βloss=[rand(multiplie) for k in 1:npopulation ]
    r=[]
    v=collect(1:npopulation)
    vanillarollout=100
    λ=0
    for i = 1:iteration

        println("iteration: $i")


        # test_position =batchedSelfplay(
        #     rollout,
        #     samplesNumber,
        #     nbworkers,
        #     trainingnets[v[end]],
        #     temptresh,
        #     λ,
        #     cpuct,
        #     value,
        #     nothing,
        # )
        indice=i
        test_position=recoverData(indice,indice,"Reversi")
        r = vcat(r, test_position)

        if length(r) > bufferSize
            r = r[length(r)-bufferSize:end]
        end
        # if i%2==0
        #     #rollout=min(rollout+50,800)
        #     t=min(t+1,20)
        # end
        if (i % 1 == 0 || train)
            for k in 1:npopulation

                if Sys.free_memory() / 2^20 < 700
                    println("memory reclaim")
                    GC.gc()
                    CuArrays.reclaim()
                end

                traininPipe(
                    batchsize,
                    trainingnets[k],
                    r,
                    epoch = epoch,
                    lr = lr[k],
                    value = value,
                    βloss = βloss[k]
            )
            end
        end

        if i % chkfrequency == 0
            if Sys.free_memory() / 2^20 < 700
                println("memory reclaim")
                GC.gc()
                CuArrays.reclaim()
            end
            p = deepcopy(test_position)
            if Sys.free_memory() / 2^20 < 700
                println("memory reclaim")
                GC.gc()
                CuArrays.reclaim()
            end
            tag = i % 20
            @save pwd() * "/Games/" * game * "/Data/data$i.json" p
            p = []
        end

        if i % 1 == 0
            scores=zeros(8)
            for j in 1:7
                for k in j+1:8

                    duel = pitNetwork(trainingnets[k], trainingnets[j], 40,temptresh,usepuct=false)
                    scores[j]+=duel[1]-duel[3]
                    scores[k]+=duel[3]-duel[1]
                end


            end
            # if i % 1 == 0
            #     duel2 = pitNetwork(
            #         net,
            #         trainingnet,
            #         50,
            #         temptresh,
            #         vanille = (true, vanillarollout),
            #         usepuct=true
            #     )
            #     print("résultat contre mcts $vanillarollout", duel2)

            # end
            println("résultat du duel: ", scores)
            v=sortperm(scores)
            duel=pitNetwork(trainingnets[end], trainingnets[v[end]], 40, temptresh,vanille=(true,vanillarollout),usepuct=false)
            println("contre mcts $vanillarollout",duel)
            if duel[1]>duel[3]
                vanillarollout += 100
            end
            for k in 1:4
                trainingnets[v[k]]=deepcopy(trainingnets[v[end]])
                lr[v[k]]=rand(multiplie)*lr[v[end]]
                βloss[v[k]]=rand(multiplie)*lr[v[end]]
            end
            println("meilleurs params so far, lr=",lr[v[end]]," βloss ",βloss[v[end]])
        end
        #
        #     if duel[1] > duel[3]
        #         net = deepcopy(trainingnet)
        #     end
        #
            reseau = trainingnets[v[end]] |> cpu
            if i % chkfrequency == 0
                if Sys.free_memory() / 2^20 < 700
                    println("memory reclaim")
                    GC.gc()
                    CuArrays.reclaim()
                end

                @save pwd() * "/Games/" * game * "/Data/reseau$i.json" reseau
            end

        if i >= 1
            bufferSize = min(500000, Int(round(bufferSize * 1.1)))
        end
    end
    #return trainingnets[v[end]]
end
