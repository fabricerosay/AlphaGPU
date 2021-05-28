using Luxor

const Columns=collect('A':'Z')

function generate_dict()
    DC=Dict()
    DCI=Dict()
    for c in 1:NN
        col=div(c-1,N)+1
        row=c-N*(col-1)
        coup=string(Columns[col])*string(row)
        DC[c]=coup
        DCI[coup]=c
    end
    DC,DCI
end

const Dic_coups,Dic_coups_inv=generate_dict()

function testvsordi(actor,readout,joueur=1;pos=nothing)

    if pos==nothing
        game=mcts_gpu.Position()
    else
        game=pos
    end

   history=[]
   while !mcts_gpu.isOver(game)[1]
       if game.player==joueur

           _,p=mcts_gpu.mcts_single([game for k in 1:1],actor,readout,8,training=false)
#
            c=argmax(p[1,:])


           println( "coups du joueur: ",Dic_coups[c])#(x,y), "\n ")



       else
           play=-1
           print("coups d internet","\n")
            while play==-1
                play=readline()
                #x=play%10
                #y=div(play-x,10)
               # if !(play in lp)
               #       play=-1
               #       println("coup non valide")
               #  end
               #c=9*x+y+1
               c=Dic_coups_inv[play]
            end



       end
       push!(history,c)
       game=mcts_gpu.play(game,c)
       display(dr(game))


   end
   return history
end

function dr(b)
    cpt=1
    radius=20
    δ=sqrt(3)/2*radius
    offset=(0,0)
    centers=[]
    for j in 0: N
        for i in 0:N
            push!(centers,Point(2*j*δ+i*δ-radius*N,i*radius*3/2-radius*N))
        end
    end
    letters=collect('A':'Z')
    dessin=@draw begin
    for k in 0:N
        for i in 1:N+1
                x=(N+1)*(i-1)+k+1
                red=(b.bplayer[x] && b.player==1) | (b.bopponent[x] && b.player==-1)
                blue=(b.bplayer[x] && b.player==-1) | (b.bopponent[x] && b.player==1)
                if red || (i==1 && k!=0)
                    sethue("red")
                elseif blue || (i!=1 && k==0)
                    sethue("blue")
                else
                    sethue("grey")
                end

                p = centers[x]
                ngon(p, radius-1, 6, π/2, :fillstroke)
                sethue("black")
                ngon(p, radius, 6, π/2, :stroke)
                sethue("white")
                if i>1 && k==0
                    text(string(letters[i-1]), p, halign=:center)
                end
                if i==1 && k>0
                    text(string(k), p, halign=:center)
                end
                cpt+=1
        end
    end
end N*80 N*80
end

function gen_mask()
    masks=[]
    mask=Hex.empty
    for j=1:N
        for k in 1:j
            mask=Bitboard.setindex(mask,true,j,k)
            mask=Bitboard.setindex(mask,true,k,j)
        end
        push!(masks,mask)
    end
    masks
end
function tisOver(pos)
	a=pos.bopponent
    masks=gen_mask()
	for j in 1:2*N-2
        display(dr(Hex.Position(a,Hex.empty,1)))
		b=Bitboard.up(a)
		c=Bitboard.right(b)
		a=Bitboard.down((a&(b|c))|(b&c))
        # for k in 3+j:N+1
        #     a=Bitboard.setindex(a,true,1,k)
        # end
        #a=a&masks[N+2-j]
        # for k in 1:j
        #     a=Bitboard.left(Bitboard.right(a))
        #     a=Bitboard.up(Bitboard.down(a))
        # end

        println(j)
		parse(Int,readline())
	end
    display(dr(Hex.Position(a,Hex.empty,1)))
end
