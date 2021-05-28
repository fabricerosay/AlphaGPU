module Hex
export Position, canPlay,play,isOver,affiche,VectorizedState,maxActions,maxLengthGame
using ..Bitboard
using .Main:NN,N



const VectorizedState=(N+1)*(N+1)
const maxActions=NN
const maxLengthGame=NN




struct Position
	bplayer::bitboard{2}
	bopponent::bitboard{2}
	player::Int8
end
const empty=bitboard{2}(N+1,N+1)
function init()
	startx=empty
	starto=empty
	for i in 3:N+1
		startx=Bitboard.setindex(startx,true,i,1)
		starto=Bitboard.setindex(starto,true,1,i)
	end
	startx,starto
end

const startx,starto=init()

Position()=Position(startx,starto,1)

function canPlay(pos,col)
	x=div(col-1,N)
	y=col-N*x
	newcol=(N+1)*(x+1)+y+1
	return ~pos.bplayer[newcol] & ~pos.bopponent[newcol]
end


function  play(pos,col)
	x=div(col-1,N)
	y=col-N*x
	newcol=(N+1)*(x+1)+y+1
	bplayer=Bitboard.setindex(pos.bplayer,true,newcol)
	return Position(pos.bopponent,bplayer,-pos.player)
end


function isOver(pos)
	a=pos.bopponent
	for j in 1:2*N-2
		b=Bitboard.up(a)
		c=Bitboard.right(b)
		a=Bitboard.down((a&(b|c))|(b&c))
		if pos.player==1
        	for k in 3+j:N+1
            	a=Bitboard.setindex(a,true,1,k)
        	end
		end
	end
	return a[N+1,N+1],-pos.player
end

function affiche(pos::Position)
	bb=pos.bplayer
	bo=pos.bopponent
	b=Array{String}(undef,(size(bb)[1],2*size(bb)[2]))
	fill!(b,"")
	if pos.player==1
		sp="X\\"
		so="O\\"
	else
		sp="O\\"
		so="X\\"
	end
	for x in 1:size(bb)[1]
		for y in 1:x+size(bb)[2]-1
			if y<x
				print(" ")
			else
				if bb[x,y-x+1]
					print(sp)
				#b[x,y]=sp
				elseif bo[x,y-x+1]
					print(so)
				#b[x,y]=so
				else
					print(" \\")
				end
			end
		end
		print("\n")
	end
end

function test(p)
	pos=Position()
	for c in p
		pos=play(pos,c)
	end
	pos
end



end
