function replay(h)
    pos=mcts_gpu.Position()
    for c in h
        pos=mcts_gpu.play(pos,c)
    end
    pos
end

function testvsordi(actor,readout,joueur=1;pos=nothing)
    if pos==nothing
        game=mcts_gpu.Position()
    else
        game=pos
    end

   history=[]
   puct=FMCTS.MctsContext(1.5,convert_back_cpu(actor),zeros(Float32,2*NN))
   while !mcts_gpu.isOver(game)[1]
       if game.player==joueur

           p,v=puct(game,readout)
           _,α,β,cc=actor(CuArray{Float32}(FMCTS.decode_cpu(game)),training=true)
  #_,p=mcts_gpu.mcts_single(actor,readout,256,vnodes,vnodesStats,leaf,newindex,1,training=false,cpuct=1.5,noise=1/36)
            # play=argmax(p)
             #println("coup:",dic_coups_inverse[play])
             println("situation: $v")
             println("α:$α, β:$β")
            c=argmax(p)

           y=div(c-1,N)
           x=(c-1)%N

           println( "coups du joueur: ",(x,y), "\n ")



       else
           play=-1
           print("coups d internet","\n")
            while play==-1
                play=parse(Int,readline())
                y=play%10
                x=div(play-y,10)
               # if !(play in lp)
               #       play=-1
               #       println("coup non valide")
               #  end
               c=N*x+y+1
            end



       end
       push!(history,c)
       game=mcts_gpu.play(game,c)
       mcts_gpu.affiche(game)


   end
   return history
end


struct ar{T}
    a::Vector{T}
end

ar{Float32}()=ar{Float32}(zeros(Float32,9))

function alphabeta(game,a,b,depth)
    w,s=winner(game)
    if w
        return game.player*s
    elseif depth==0
        return 0.1
    else
        best=-Inf
        for action in 1:getActionNumber(game)
            child=playIndex(game,findIndex(game,action))
            v=-alphabeta(child,-b,-a,depth-1)
            v=-v
            if v>best
                best=v
                if best>a
                    a=best
                    if a>=b
                        return a
                    end
                end
            end
        end
    end

    return a
end

function alphabeta(game,depth)
    lp=legal_plays(game)
    best=-Inf
    bestmove=[]
    for play in lp
        v=-alphabeta(next_state!(game,play),-Inf,Inf,depth)
        if v>=best
            best=v
            push!(bestmove,play)
        end
    end
    bestmove,best
end
