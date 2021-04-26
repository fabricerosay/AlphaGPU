
function testvsordi(actor,readout,joueur=1)
   game=GameEnv(1)



   root=game
   history=[]

   puct=MctsContext(0,1.5,4,actor)
   while !winner(game)[1]
       if game.player==joueur

           pla,v=puct([game],readout,true)
#
            play=argmax(pla)-1

           coup=play+1

           println( "coups du joueur: ",coup, "\n ")

           println("situation: ",v)
           push!(history,game)

       else
           lp=legalPlays(game)
           println(lp)
           play=-1
           print("coups d internet","\n")
            while play==-1
                play=parse(Int,readline())-1

               if !(play in lp)
                     play=-1
                     println("coup non valide")
                end

            end



       end

       game=playIndex(game,play)



   end
   println("winner: ",winner(game)[2])
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
