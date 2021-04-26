module Game
using DataStructures
const N=7
const NN=6*7
const sizeInput=(6,7,5)
const bitarray_size=84+10*7+2
const maxActions=7
using StaticArrays
export GameEnv,playIndex,findIndex,simul,winner,getActionNumber,getAction,getNumber,legalPlays,
decoder,N,NN,maxActions,sizeInput,bitarray_size,encoder
using Random

const  WIDTH = 7
const  HEIGHT = 6
const  MIN_SCORE = -(WIDTH*HEIGHT)/2 + 3;
const  MAX_SCORE = (WIDTH*HEIGHT+1)/2 - 3;
const columnOrder=[3,2,4,1,5,0,6]

struct Position
	mask::UInt64
	current_position::UInt64
end

Position()=Position(0,0)

struct GameEnv
  state :: Tuple{Position,Position,Position}
  player :: Int
  round:: Int8
end

GameEnv(n)=GameEnv((Position(),Position(),Position()),n,0)
GameEnv()=GameEnv(1)

function canPlay(pos,col)
	  return (pos.mask & top_mask(col)) == 0
end


function  play(pos,col)
	  current_position = pos.mask ⊻ pos.current_position
	  mask = (pos.mask + bottom_mask(col)) | pos.mask
	  return Position(mask,current_position)
end


function isWinningMove(pos,col)
	  position = pos.current_position;
	  position |= (pos.mask + bottom_mask(col)) & column_mask(col);
	  return alignment(position);
end


function key(pos)
	  return pos.current_position + pos.mask
end


function  alignment(pos)

	   m = pos & (pos >> (HEIGHT+1))
	  if (m & (m >> (2*(HEIGHT+1))))!=0

		  return true
	  end


	  m = pos & (pos >> HEIGHT)
	  if (m & (m >> (2*HEIGHT)))!=0

		  return true
	  end


	  m = pos & (pos >> (HEIGHT+2))
	  if (m & (m >> (2*(HEIGHT+2))))!=0

		  return true
	  end

	  m = pos & (pos >> 1)
	  if (m & (m >> 2))!=0

		  return true
	  end
	  return false
  end


function top_mask(col)

	return (UInt64(1) << (HEIGHT - 1)) << (col*(HEIGHT+1))
end

function  bottom_mask(col)
	  return UInt64(1) << (col*(HEIGHT+1))
end


function column_mask(col)
	  return ((UInt64(1) << HEIGHT)-1) << (col*(HEIGHT+1))
end

function getActionNumber(game)
	moves=0
    pos=game.state[1]
	for col in 0:6
		if canPlay(pos,col)
			moves+=1
		end
	end
	moves
end

function findIndex(game,k::Int)
    j=-1
    cpt=0
    while cpt<k
        j+=1
        if canPlay(game.state[1],j)
            cpt+=1
        end
    end
    return j
end
function winner(game)
    pos=game.state[1]
    current_position = pos.mask ⊻ pos.current_position
    if alignment(current_position)
        return true,-game.player
    elseif getActionNumber(game)==0
        return true,0
    else
        return false,0
    end
end

function playIndex(game::GameEnv,k)
	state=play(game.state[1],k)
	return GameEnv((state,game.state[1],game.state[2]),-game.player,game.round+1)
end

legalPlays(game)=[i for i in 0:6 if canPlay(game.state[1],i)]
getNumber(n)=n+1
getAction(n)=n-1

function simul(game::GameEnv)
    game_=game
    w,r=winner(game_)
    while !w
        play=findIndex(game_,rand(1:getActionNumber(game)))
        game_=playIndex(game_,play)
        w,r=winner(game_)

    end
    return (1+r*game.player)/2
end



@inline @inbounds function simul(game::GameEnv,n)
    s=0
    for i in 1:n
        s+=simul(game)
    end
    return s/n

end



function simul()
    game=GameEnv()
    simul(game)
end


function decoder(game,answer=nothing)
	if answer==nothing
		answer=falses(98)
	end
		position=game.state[1]
		pos=position.current_position
		mask=position.mask
		for j in 0:48

			answer[j+1]=(pos>>j)&UInt(1)
			answer[j+50]=((pos⊻mask)>>j)&UInt(1)

		end

	answer
end


end

using .Game
import .Game:playIndex,getActionNumber,legalPlays
