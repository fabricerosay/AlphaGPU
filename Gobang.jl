module GoBang
export Position, canPlay,play,isOver,affiche,VectorizedState,maxActions,maxLengthGame
using ..Bitboard


const N=3
const NN=N*N
const VectorizedState=NN
const maxActions=NN
const maxLengthGame=NN
const Nvict=3



struct Position
	bplayer::bitboard{2}
	bopponent::bitboard{2}
	player::Int8
end

Position()=Position(bitboard{2}(N,N),bitboard{2}(N,N),1)

function canPlay(pos,col)
	  return ~pos.bplayer[col] & ~pos.bopponent[col]
end


function  play(pos,col)
	bplayer=Bitboard.setindex(pos.bplayer,true,col)
	return Position(pos.bopponent,bplayer,-pos.player)
end


function isOver(pos)
	board=pos.bopponent
    for j in 1:Nvict-1
		board=board&right(board)
	end
	if num_bit(board)!=0
		return true,-pos.player
	end

	board=pos.bopponent
    for j in 1:Nvict-1
		board=board&down(board)
	end
	if num_bit(board)!=0
		return true,-pos.player
	end

	board=pos.bopponent
    for j in 1:Nvict-1
		board=board&down(right(board))
	end
	if num_bit(board)!=0
		return true,-pos.player
	end

	board=pos.bopponent
    for j in 1:Nvict-1
		board=board & left(down(board))
	end
	if num_bit(board)!=0
		return true,-pos.player
	end

	return num_bit(pos.bplayer)+num_bit(pos.bopponent)==NN,Int8(0)
end

function affiche(pos::Position)
	bb=pos.bplayer
	bo=pos.bopponent
    b=Array{Char}(undef,size(bb))
	fill!(b,' ')
	if pos.player==1
		sp='X'
		so='O'
	else
		sp='O'
		so='X'
	end
    for x in 1:size(bb)[1], y in 1:size(bb)[2]
		if bb[x,y]
        	b[x,y]=sp
		elseif bo[x,y]
			b[x,y]=so
		end
    end
    display(b)
end

end
