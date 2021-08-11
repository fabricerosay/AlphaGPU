module GoBang
export Position, canPlay,play,isOver,affiche,VectorizedState,FeatureSize,maxActions,maxLengthGame
using ..Bitboard
using .Main:NN,N,Nvict



const VectorizedState=NN
const FeatureSize=NN
const maxActions=NN
const maxLengthGame=NN




struct Position
	bplayer::bitboard{2}
	bopponent::bitboard{2}
	player::Int8
	round::Int8
end

Position()=Position(bitboard{2}(N,N),bitboard{2}(N,N),1,0)

function canPlay(pos,col)
	  return ~pos.bplayer[col] & ~pos.bopponent[col]
end


function  play(pos,col)
	bplayer=Bitboard.setindex(pos.bplayer,true,col)
	return Position(pos.bopponent,bplayer,-pos.player,pos.round+1)
end


function isOver(pos)
	board=pos.bopponent
    for j in 1:Nvict-1
		board=board&right(board)
	end
	if num_bit(board)!=0
		return true,-pos.player*(Int8(NN+1)-pos.round)
	end

	board=pos.bopponent
    for j in 1:Nvict-1
		board=board&down(board)
	end
	if num_bit(board)!=0
		return true,-pos.player*(Int8(NN+1)-pos.round)
	end

	board=pos.bopponent
    for j in 1:Nvict-1
		board=board&down(right(board))
	end
	if num_bit(board)!=0
		return true,-pos.player*(Int8(NN+1)-pos.round)
	end

	board=pos.bopponent
    for j in 1:Nvict-1
		board=board & left(down(board))
	end
	if num_bit(board)!=0
		return true,-pos.player*(Int8(NN+1)-pos.round)
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
