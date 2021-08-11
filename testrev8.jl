letters=["a","b","c","d","e","f","g","h"]
dic_coups=Dict()
dic_coups_inverse=Dict()
for case in 1:8
    for zone in 1:8

        dic_coups[letters[case]*"$zone"]=8*(case-1)+zone
        dic_coups_inverse[(zone)+8*(case-1)]=letters[case]*"$zone"
    end
end
dic_coups["p"]=65
dic_coups_inverse[65]="pass"
function testvsordi(actor,readout,player=-1)
  game=RevSix.Position()
 # vnodes,vnodesStats,leaf,newindex=mcts_gpu.init(1,readout)
puct=FMCTS.MctsContext(1.5,convert_back_cpu(actor),zeros(Float32,128))
   while !RevSix.isOver(game)[1]
       if game.player==player
         p,v=puct(game,readout)
         _,α,β,cc=actor(CuArray{Float32}(FMCTS.decode_cpu(game)),training=true)
#_,p=mcts_gpu.mcts_single(actor,readout,256,vnodes,vnodesStats,leaf,newindex,1,training=false,cpuct=1.5,noise=1/36)
           play=argmax(p)
           println("coup:",dic_coups_inverse[play])
           println("situation: $v")
           println("α:$α, β:$β")
       else

           play=-1
           print("coups d internet","\n")
            while play==-1

                   play=readline()


            end

            play=dic_coups[play]
            println(play)



       end
       #root=root.child[tc[play]]
       game=RevSix.play(game,play)
#       mcts_gpu.re_init(cu([game for k in 1:1]),vnodes,1,1,1)



   end
   w=RevSix.isOver(game)[2]
   if w>0
       println("winner: puct")
   elseif w==0
       println("match nul")
   else
       println("tothello")
   end

end
