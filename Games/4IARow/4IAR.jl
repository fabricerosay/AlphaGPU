const N=7
const NN=6*7

const maxActions=7
using StaticArrays

struct GameEnv
  state :: SArray{Tuple{6,7,2}}
  line_state:: SVector{7,Int}
  index::Int
  nb_coups::Int
  player :: Int
end

GameEnv(n)=GameEnv(zeros(Int,(6,7,2)),6*ones(Int,7),-1,7,n)
GameEnv()=GameEnv(1)

function playIndex(game::GameEnv, index::Int)
    if game.player==1
        player=0
    else
        player=1
    end
    line=game.line_state[index]
    if line==1
        nb_coups=game.nb_coups-1
    else
        nb_coups=game.nb_coups
    end
    line_state=setindex(game.line_state,game.line_state[index]-1,index)
    state=setindex(game.state,1,(index-1)*6+line+42*player)
    return GameEnv(state,line_state,index,nb_coups,-game.player)
end

function findIndex(game::GameEnv,k::Int)
    j=0
    cpt=0
    while cpt<k
        j+=1
        if game.line_state[j]>0
            cpt+=1
        end
    end
    return j
end

function CheckLine(game::GameEnv,index::Int)
    xs=game.line_state[index]+1
    ys=1
    s=0
    v=false
    if game.player==1
        player=2
    else
        player=1
    end
    while ys<=7
        t=game.state[xs,ys,player]
        s=s*t+t
        if s==4
            v=true
            break
        end
        ys+=1
    end
    if v
        return -game.player
    else
        return -2
    end
end

function CheckColonne(game::GameEnv,index::Int)
    xs=game.line_state[index]+1
    ys=index
    s=0
    v=false
    if game.player==1
        player=2
    else
        player=1
    end
    while xs<=6
        t=game.state[xs,ys,player]
        s=s*t+t
        if s==4
            v=true
            break
        end
        xs+=1
    end
    if v
        return -game.player
    else
        return -2
    end
end


function CheckDiagg(game::GameEnv,index::Int)
    xs=game.line_state[index]+1
    ys=index
    kmax=min(3,7-ys,6-xs)
    kmin=max(4-xs,4-ys)
    if game.player==1
        player=2
    else
        player=1
    end
    if kmin>kmax
        return -2
    else
        for k in kmin:kmax
            v=true
            j=0
            while j<=3
                v&=game.state[xs+k-j,ys+k-j,player]==1
                j+=1
            end
            if  v
                return -game.player
            end
        end

    end

    return -2

end


function CheckDiagd(game::GameEnv,index::Int)
    xs=game.line_state[index]+1
    ys=index
    kmax=min(3,7-ys,xs-1)
    kmin=max(0,xs-3,4-ys)
    if game.player==1
        player=2
    else
        player=1
    end
    if kmin>kmax
        return -2
    else
        for k in kmin:kmax
            v=true
            j=0
            while j<=3
                v&=game.state[xs-k+j,ys+k-j,player]==1
                j+=1
            end
            if  v
                return -game.player
            end
        end

    end

    return -2
end

function winner(game::GameEnv,index::Int)
        if index==-1
            return false,0
        end
        test=CheckLine(game,index)
        if test!=-2
            return true,test
        end
        test=CheckColonne(game,index)
        if test!=-2
            return true,test
        end
        test=CheckDiagg(game,index)
        if test!=-2
            return true,test
        end
        test=CheckDiagd(game,index)
        if test!=-2
            return true,test
        end
        return false,0
end

function winner(game::GameEnv)
    r,w=winner(game,game.index)
    if r
        return r,w
    elseif game.nb_coups==0
        return true,0
    else
        return false,0
    end
end

@inline @inbounds function getActionNumber(game::GameEnv)
    return game.nb_coups
end

function decoder(game::GameEnv)
    answer=zeros(6,7,2,1)
    if game.player==1
        answer[:,:,:,1].=game.state
    else
        answer[:,:,1,1].=game.state[:,:,2]
        answer[:,:,2,1].=game.state[:,:,1]
    end
    return answer
end

function symstate(state)
    answer=zeros(6,7,2,1)
    for i in 1:6
        for j in 1:7
            answer[i,j,1,1]=state[i,8-j,1,1]
            answer[i,j,2,1]=state[i,8-j,2,1]
        end
    end
    return answer
end

function sympo(po)
    answer=zeros(7)
    for i in 1:7
        answer[i]=po[8-i]
    end
    return answer
end

function simul(game::GameEnv)
    game_=game
    w,r=winner(game_)
    while !w
        play=findIndex(game_,rand(1:game_.nb_coups))
        game_=playIndex(game_,play)
        w,r=winner(game_)

    end
    return r*game.player
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

function simul()
    game=GameEnv()
    simul(game)
end


function deroule()
    _,l=simul()
    for k in l
        println(k)
        while readline()!=""
        end
    end
end
