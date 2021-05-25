module FourIARow
export Position, canPlay,play,isOver,affiche,VectorizedState,maxActions,maxLengthGame
using ..Bitboard


const Height=6
const Width=7
const VectorizedState=42
const maxActions=7
const maxLengthGame=42
const Nvict=4



struct Position
	bplayer::bitboard{2}
	bopponent::bitboard{2}
	player::Int8
end

Position()=Position(bitboard{2}(Height,Width),bitboard{2}(Height,Width),1)

function canPlay(pos,col)
	  return ~pos.bplayer[1,col] & ~pos.bopponent[1,col]
end


function  play(pos,col)
	free=1
	lastfree=1
	empty=~(pos.bplayer|pos.bopponent)
	for i in 1:Height
		if empty[i,col]
			free=i
		else
			break
		end
	end
	c=Height*(col-1)+free
	bplayer=Bitboard.setindex(pos.bplayer,true,c)
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

	return num_bit(pos.bplayer)+num_bit(pos.bopponent)==maxLengthGame,Int8(0)
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
