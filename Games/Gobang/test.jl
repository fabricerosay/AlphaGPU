
dic_coups=Dict()
dic_coups_inverse=Dict()
for k in 1:NN
    dic_coups_inverse[k]=(div(k-1,N),(k-1)%N)
    i=div(k-1,N)
    j=(k-1)%N
    dic_coups["$i$j"]=k
end
function testvsordi(readout)
   game=GameEnv(1)
  # root=newnode(game,-1,-1)

   #w=0
   root=game
   while !winner(game)[1]
       if game.player==1



            pla,v=puct(game,readout)#utcsearch(game,readout,actor)#mcts(game,readout,pool,true,verbose=true,c=1.1)
            play=getAction(argmax(pla))
            #v=0
            #v=maximum(play)
           coup=dic_coups_inverse[play]
           println( "coups du joueur: ",coup, "\n ")
           println("nombre", play)
           println("situation: ",v)


       else
           lp=legalPlays(game)
           play=-1
           print("coups d internet","\n")
            while play==-1
               while !haskey(dic_coups,play)
                   play=readline()
               end
               if !(dic_coups[play] in lp)
                     play=-1
                     println("coup non valide")
                end

            end

            play=dic_coups[play]
            # index=1
            # for el in lp
            #     if el==play
            #         break
            #     end
            #     index+=1
            # end
       end
       #root=root.child[tc[play]]
       game=playIndex(game,play)



   end
end

@inline function simul2(game::GameEnv)
win=0
lose=0
for i in 1:100
    current=game
    finished = results(current)
  while finished==-2


     # game.currentPlayer == 1, genmove(game, bot1), genmove(game, bot2)
        p=mcts(current,0.1,nothing,true;c=1.2)[1]
         current = playIndex(current, )

    finished = results(current)
end
if finished==1
    win+=1
elseif finished==2
    lose+=1
end
end
return win,lose
end
