const N=9
const NN=N*N
maxActions=NN


struct GameEnv
    state::SVector{81,Int}
    miniState::SVector{9,Int}
    freeMovesZone::SVector{9,Int}
    zone::Int
    player::Int
end


GameEnv(player)=GameEnv(zeros(81),zeros(9),9*ones(9),-1,player)

GameEnv()=GameEnv(1)

function decoder(game)
    answer=zeros(9,9,6,1)
    player=game.player
    for k in 1:81
        zone=div(k-1,9)
        case=(k-1)%9
        zx=div(zone,3)
        zy=zone%3
        cx=div(case,3)
        cy=case%3
        x=3*zx+cx+1
        y=3*zy+cy+1

        if game.state[k]==game.player
            answer[y,x,1,1] = 1#game.player*game.state[k]
        elseif game.state[k] == -game.player
            answer[y,x,2,1] = 1
        end
        if game.miniState[zone+1]==game.player
            answer[y,x,3,1] = 1#game.player*game.state[k]
        elseif game.miniState[zone+1] == -game.player
            answer[y,x,4,1] = 1
        end
        if game.state[k]==0 && (zone==game.zone-1 || game.zone==-1)
            answer[y,x,5,1]=1
        end
    end
    answer[:,:,6,1].=1
return answer
end
#  function decoder(game)
#     answer=zeros(189)
#
#     player=game.player
#     for k in 1:81
#         if game.state[k]==player
#             answer[k]=1
#         elseif game.state[k]==-player
#             answer[k+81]=1
#         end
#     end
#     for k in 1:9
#         if game.miniState[k]==player
#             answer[k+162]=1
#         elseif game.miniState[k]==-player
#             answer[k+171]=1
#         elseif game.freeMovesZone[k]>0 && (game.zone==-1 || game.zone==k)
#             answer[k+180]=1
#         end
#     end
#
# return answer
#  end

@inline @inbounds function winner(b, zone,moves)
  if b[1+zone] != 0 && b[1+zone] == b[2+zone] == b[3+zone]
    true, b[1+zone]
  elseif b[4+zone] != 0 && b[4+zone] == b[5+zone] == b[6+zone]
    true, b[4+zone]
  elseif b[7+zone] != 0 && b[7+zone] == b[8+zone] == b[9+zone]
    true, b[7+zone]
  elseif b[1+zone] != 0 && b[1+zone] == b[4+zone] == b[7+zone]
    true, b[1+zone]
  elseif b[2+zone] != 0 && b[2+zone] == b[5+zone] == b[8+zone]
    true, b[2+zone]
  elseif b[3+zone] != 0 && b[3+zone] == b[6+zone] == b[9+zone]
    true, b[3+zone]
  elseif b[1+zone] != 0 && b[1+zone] == b[5+zone] == b[9+zone]
    true, b[1+zone]
  elseif b[3+zone] != 0 && b[3+zone] == b[5+zone] == b[7+zone]
    true, b[3+zone]
  elseif moves == 0  # no winner, no empty cell -> draw
    true, 0
  else  # no winner, empty cell -> game not finished
    false, 0
  end
end

@inline @inbounds function winner(game::GameEnv)
    return winner(game.miniState,0,sum(game.freeMovesZone))
end

@inline @inbounds function playIndex(game::GameEnv,move::Int)
    state=setindex(game.state,game.player,move)

    zone=div(move-1,9)+1
    @assert (zone==game.zone || game.zone==-1)
    if game.freeMovesZone[zone]==0
        print("faute")
    end
    f,r=winner(state,9*(zone-1),game.freeMovesZone[zone]-1)
    if f
        miniState=setindex(game.miniState,r,zone)
        freeMovesZone=setindex(game.freeMovesZone,0,zone)
    else
        miniState=game.miniState
        freeMovesZone=setindex(game.freeMovesZone,game.freeMovesZone[zone]-1,zone)
    end
    case=(move-1)%9+1
    if freeMovesZone[case]==0
        zone=-1
    else
        zone=case
    end
    return GameEnv(state,miniState,freeMovesZone,zone,-game.player)
end

@inline @inbounds function getActionNumber(game::GameEnv)
    zone=game.zone
    if zone==-1
        return sum(game.freeMovesZone)
    else
        return game.freeMovesZone[zone]
    end
end

potentiallyPlayable(i,game)=(game.state[i]==0 && game.freeMovesZone[div(i-1,9)+1]!=0)

@inline @inbounds function findIndex(game::GameEnv,k::Int)
    zone=game.zone
    if zone==-1
        start=1
        finish=81
    else
        start=9*(zone-1)+1
        finish=start+8
    end
    cpt=0
    for i in start:finish
        if potentiallyPlayable(i,game)
            cpt+=1
            if cpt==k
                return i
            end
        end
    end

end


@inline @inbounds function simul(game::GameEnv=GameEnv(1))
    game_=game
    while !winner(game_)[1]
        moveindex=findIndex(game_,rand(1:getActionNumber(game_)))
        game_=playIndex(game_,moveindex)
    end
        winner(game_)[2]*game.player
end



legalPlays(game::GameEnv)=[findIndex(game,k) for k in 1:getActionNumber(game)]


@inline @inbounds function simul(game::GameEnv,n)
    s=0
    for i in 1:n
        s+=simul(game)
    end
    return s/n
end

getNumber(n)=n

getAction(n)=n
