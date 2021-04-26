const N=5
const NN=N*N
const nvict=4

const maxActions=NN

mutable struct Board
    p1::Array{Int,2}
    p2::Array{Int,2}
end

mutable struct GameEnv
    state:: Board
    nbFreeMoves::Int
    player::Int
    lastmove::Int
end

GameEnv() = GameEnv(Board(zeros(Int,(N,N)),zeros(Int,(N,N))) , NN, 1,0)
GameEnv(player)= GameEnv(Board(zeros(Int,(N,N)),zeros(Int,(N,N))) , NN, player,0)


@inline @inbounds function findIndex(b::Array{Int,2},k::Int)
    cpt=0
    for j in 1:NN
        if b[j]==0
            cpt+=1
            if cpt==k
                return j
            end
        end
    end
    return k
end

function findIndex(game::GameEnv,k::Int)
    return findIndex(game.state.p1+game.state.p2,k)
end

@inline @inbounds function playIndex(game_::GameEnv, move::Int)
    game=deepcopy(game_)
    if game.player==1
        game.state.p1[move]=1
    else
        game.state.p2[move]=1
    end
    game.player=-game.player
    game.nbFreeMoves-=1
    game.lastmove=move
    return game
end

legalPlays(g::GameEnv)=[findIndex(g,k) for k in 1:getActionNumber(g)]

@inline @inbounds function winner_dd(game::Array{Int,2},move)
    if move==0
        return false
    end
    cpt=0
    j=div(move-1,N)+1
    i=move-N*(j-1)
    kmin=max(1-i,1-j,-(nvict-1))
    kmax=min(N-i,N-j,(nvict-1))

    for k in kmin:kmax
        cpt=(cpt+1)*game[i+k,j+k]
        if cpt==nvict
            return true
        end
    end
    cpt=0
    kmin=max(1-i,j-N,-(nvict-1))
    kmax=min(N-i,j-1,(nvict-1))
    for k in kmin:kmax
        cpt=(cpt+1)*game[i+k,j-k]

        if cpt==nvict
            return true
        end
    end
    return false
end

@inline @inbounds function winner_hv(game::Array{Int,2},move)
    if move==0
        return false
    end
    cpt=0
    j=div(move-1,N)+1
    i=move-N*(j-1)

    for k in max(i-(nvict-1),1):min(i+(nvict-1),N)
        cpt=(cpt+1)*game[k,j]
        if cpt==nvict
            return true
        end
    end
    cpt=0
    for k in max(j-(nvict-1),1):min(j+(nvict-1),N)
        cpt=(cpt+1)*game[i,k]
        if cpt==nvict
            return true
        end
    end
    return false
end


@inline @inbounds function winner(game::GameEnv)
    if game.player==1
        if winner_hv(game.state.p2,game.lastmove)||winner_dd(game.state.p2,game.lastmove)
            return true,-1
        else
            return game.nbFreeMoves==0,0
        end
    else
        if winner_hv(game.state.p1,game.lastmove)||winner_dd(game.state.p1,game.lastmove)
            return true,1
        else
            return game.nbFreeMoves==0,0
        end
    end
end


@inline @inbounds function getActionNumber(game)
        return game.nbFreeMoves
end

function decoder(game::GameEnv)
    answer=zeros(Float32,(N,N,3,1))
    if game.player==1
        answer[:,:,1,1].=game.state.p1
        answer[:,:,2,1].=game.state.p2
    else
        answer[:,:,1,1].=game.state.p2
        answer[:,:,2,1].=game.state.p1
    end
    answer[:,:,3,1].=1
    return answer
end
# function decoder(game::GameEnv)
#     answer=zeros(Float32,2*NN)
#     if game.player==1
#         answer[1:NN].=reshape(game.state.p1,NN)
#         answer[NN+1:2*NN].=reshape(game.state.p2,NN)
#     else
#         answer[1:NN].=reshape(game.state.p2,NN)
#         answer[NN+1:2*NN].=reshape(game.state.p1,NN)
#     end
#     return answer
# end



@inline @inbounds function simul(game::GameEnv=GameEnv())
    game_=deepcopy(game)
    while !winner(game_)[1]
        moveindex=findIndex(game_,rand(1:getActionNumber(game_)))
        game_=playIndex(game_,moveindex)
    end
    winner(game_)[2]*game.player
end


@inline @inbounds function simul(game::GameEnv,n)
    s=0
    for i in 1:n
        s+=simul(game)
    end
    return s/n
end

getNumber(n)=n

getAction(n)=n
