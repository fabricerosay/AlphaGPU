const maxActions=37
const N=6
const NN=N*N
function translate(n::UInt64)
    answer=zeros(Float32,(N,N))
    k=0
    for i in 1:N
        for j in 1:N
        answer[7-i,7-j]=(n>>k) & UInt64(1)
        k+=1
    end
    end
    return answer
end


struct Board
    player::UInt64
    opponent::UInt64
end

struct GameEnv
    board::Board
    moves::UInt64
    number_moves::Int
    player::Int
    pass::Bool
    round::Int
end

function decoder(s::GameEnv)
    answer=zeros(Float32,(N,N,3,1))
    answer[:,:,1,1].=translate(s.board.player)
    answer[:,:,2,1].=translate(s.board.opponent)
    answer[:,:,3,1].=1
    answer
end

encode(r::UInt,c::UInt)=UInt64(1)<<(N*(r-1)+c-1)
encode(rc::Tuple{Int,Int})=encode(UInt(rc[1]),UInt(rc[2]))

function Encode()
    d=zeros(UInt64,(N,N))
    for i in UInt64(1):6
        for j in UInt64(1):6
            d[i,j]=encode(i,j)
        end
    end
    return d
end

const ENCODE=Encode()

const CENTERS_BITS_w = sum(encode(i) for i in [(3, 3), (4, 4)])
const CENTERS_BITS_b = sum(encode(i) for i in [(4, 3), (3,4)])

GameEnv()=GameEnv(Board(CENTERS_BITS_b,CENTERS_BITS_w),0x0000000008402100,4,1,false,0)

GameEnv(player)=GameEnv(Board(CENTERS_BITS_b,CENTERS_BITS_w),0x0000000008402100,4,player,false,0)


@inline function bas(n::UInt64)::UInt64
    return n>>6 & 0x0000000fffffffff
end

@inline function haut(n::UInt64)::UInt64
    return n<<6 & 0x0000000fffffffff
end


@inline function droite(n::UInt64)::UInt64
    return n>>1 & 0x00000007df7df7df
end


@inline function gauche(n::UInt64)::UInt64
    return n<<1 & 0x0000000fbefbefbe
end

@inline diaghd(x::UInt64)::UInt64=haut(droite(x))

@inline diaghg(x::UInt64)::UInt64=haut(gauche(x))

@inline diagbd(x::UInt64)::UInt64=bas(droite(x))

@inline diagbg(x::UInt64)::UInt64=bas(gauche(x))


@inline function legal_play(tabjoueur::UInt64,tabadversaire::UInt64,dir)
    tabvide=~tabjoueur&~tabadversaire
    moves=UInt64(0)
    candidats=dir(tabjoueur) & tabadversaire
    while candidats!=0
        moves|=tabvide & dir(candidats)
        candidats=tabadversaire & dir(candidats)
    end
    return moves
end

@inline function legal_play(tabjoueur::UInt64,tabadversaire::UInt64)
    return legal_play(tabjoueur,tabadversaire,haut)|legal_play(tabjoueur,tabadversaire,bas)|legal_play(tabjoueur,tabadversaire,gauche)|legal_play(tabjoueur,tabadversaire,droite)|
    legal_play(tabjoueur,tabadversaire,diaghg)|legal_play(tabjoueur,tabadversaire,diagbg)|legal_play(tabjoueur,tabadversaire,diaghd)|legal_play(tabjoueur,tabadversaire,diagbd)
end

@inline function legal_play(state::GameEnv)
    return legal_play(state.board.player,state.board.opponent)
end

@inline function findIndex(moves::UInt64, k::Int)
    if moves==0
        return NN
    else
    cpt=0
    for i in 0:35
        if 1 & moves>>i !=0
            cpt+=1
            if cpt==k
                return i
            end
        end
    end
    end
    return 36
end
#
 @inline function findIndex(state::GameEnv,k::Int)
     return findIndex(state.moves,k)
 end




function legalPlays(state::GameEnv)
    if state.moves==0
        if state.pass
            return []
        else
            return [36]
        end
    else
        lp=[]
        for i in 0:35
            if state.moves>>i & 1!=0
                push!(lp,i)
            end
        end
        return lp
    end
end


function flippar(tabjoueur::UInt64,tabadversaire::UInt64,play::UInt64,dir)
    candidats=dir(play) & tabadversaire
    toflip=candidats
    while candidats!=0
        candidats=tabadversaire & dir(candidats)
        toflip|=candidats
    end
    if dir(toflip) & tabjoueur!=0
        return toflip
    else
        return UInt64(0)
    end
end

function flip(tabjoueur::UInt64,tabadversaire::UInt64,play::Int64)
    if play==36
        h=UInt64(0)
    else
        test=one(UInt64)<<play
        h=flippar(tabjoueur,tabadversaire,test,haut)
        h|=flippar(tabjoueur,tabadversaire,test,bas)
        h|=flippar(tabjoueur,tabadversaire,test,gauche)
        h|=flippar(tabjoueur,tabadversaire,test,droite)
        h|=flippar(tabjoueur,tabadversaire,test,diaghd)
        h|=flippar(tabjoueur,tabadversaire,test,diaghg)
        h|=flippar(tabjoueur,tabadversaire,test,diagbd)
        h|=flippar(tabjoueur,tabadversaire,test,diagbg)
    end
    return h
end

function playIndex(state::GameEnv,play::Int)
    if play==36
        moves=legal_play(state.board.opponent,state.board.player)
        return GameEnv(Board(state.board.opponent,state.board.player),moves,count_ones(moves),-state.player,true,state.round)
    else
        tabjoueur=state.board.player
        tabadversaire=state.board.opponent
        h=flip(tabjoueur,tabadversaire,play)
        tabjoueur⊻=h
        tabadversaire⊻=h
        tabjoueur|=(UInt64(1)<<play)
        moves=legal_play(tabadversaire,tabjoueur)
        return GameEnv(Board(tabadversaire,tabjoueur),moves, count_ones(moves),-state.player,false,state.round+1)
    end
end

function PlayIndex(state::GameEnv,play::Tuple{Int,Int})
    return PlayIndex(state,encode(play))
end



function winner(state::GameEnv)
    if state.number_moves!=0 || !state.pass
        return false,0
    else
            test=count_ones(state.board.player)-count_ones(state.board.opponent)
            if test>0
                return true,state.player
            elseif test==0
                return true,0
            else
                return true,-state.player
            end
    end
end

@inline getActionNumber(game::GameEnv)=game.number_moves==0 ? 1 : game.number_moves



@inline @inbounds function number_moves(game)
    if game.moves==0 && !game.pass
        return 1
    else
        return game.number_moves
    end
end





@inline @inbounds function simul(game::GameEnv=GameEnv())
    game_=game
    while !winner(game_)[1] && game_.round<40
        moveindex=findIndex(game_,rand(1:number_moves(game_)))
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

getNumber(n)=n+1

getAction(n)=n-1

function scoreFinal(tableau)
    return sum(tableau[:,:,1,1])-sum(tableau[:,:,2,1])
end
