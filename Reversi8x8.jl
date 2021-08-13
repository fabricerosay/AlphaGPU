module RevSix
export Position, canPlay,play,isOver,affiche,VectorizedState,FeatureSize,maxActions,maxLengthGame
using ..Bitboard

const FeatureSize=64
const VectorizedState=64
const maxActions=65
const maxLengthGame=70

const empty=bitboard{2}(8,8)
start=Bitboard.setindex(empty,true,4,5)
const starto=Bitboard.setindex(start,true,5,4)
start=Bitboard.setindex(empty,true,5,5)
const startp=Bitboard.setindex(start,true,4,4)


@inline diaghd(x)=Bitboard.up(right(x))

@inline diaghg(x)=Bitboard.up(left(x))

@inline diagbd(x)=down(right(x))

@inline diagbg(x)=down(left(x))


@inline function legal_play(tabjoueur,tabadversaire,dir)
    tabvide=~tabjoueur&~tabadversaire
    moves=empty
    candidats=dir(tabjoueur) & tabadversaire
    while num_bit(candidats)!=0
        moves|=tabvide & dir(candidats)
        candidats=tabadversaire & dir(candidats)
    end
    return moves
end

@inline function legalplay(tabjoueur,tabadversaire)
    return legal_play(tabjoueur,tabadversaire,Bitboard.up)|legal_play(tabjoueur,tabadversaire,down)|legal_play(tabjoueur,tabadversaire,left)|legal_play(tabjoueur,tabadversaire,right)|
    legal_play(tabjoueur,tabadversaire,diaghg)|legal_play(tabjoueur,tabadversaire,diagbg)|legal_play(tabjoueur,tabadversaire,diaghd)|legal_play(tabjoueur,tabadversaire,diagbd)
end



function flippar(tabjoueur,tabadversaire,play,dir)
    candidats=dir(play) & tabadversaire
    toflip=candidats
    while num_bit(candidats)!=0
        candidats=tabadversaire & dir(candidats)
        toflip|=candidats
    end
    if num_bit(dir(toflip) & tabjoueur)!=0
        return toflip
    else
        return empty
    end
end

function flip(tabjoueur,tabadversaire,play::Int64)
        test=Bitboard.setindex(empty,true,play)
        h=flippar(tabjoueur,tabadversaire,test,Bitboard.up)
        h|=flippar(tabjoueur,tabadversaire,test,down)
        h|=flippar(tabjoueur,tabadversaire,test,left)
        h|=flippar(tabjoueur,tabadversaire,test,right)
        h|=flippar(tabjoueur,tabadversaire,test,diaghd)
        h|=flippar(tabjoueur,tabadversaire,test,diaghg)
        h|=flippar(tabjoueur,tabadversaire,test,diagbd)
        h|=flippar(tabjoueur,tabadversaire,test,diagbg)

    return h
end


struct Position
	bplayer::bitboard{2}
	bopponent::bitboard{2}
	legalplay::bitboard{2}
	player::Int8
end

const lpstart=legalplay(starto,startp)

Position()=Position(starto,startp,lpstart,1)

function canPlay(pos,c)
	if c==65
		return num_bit(pos.legalplay)==0
	else
		return pos.legalplay[c]
	end
end


function  play(pos,c)
    tabjoueur=pos.bplayer
    tabadversaire=pos.bopponent
	if c==65
        moves=legalplay(tabadversaire,tabjoueur)
        return Position(pos.bopponent,pos.bplayer,moves,-pos.player)
    end
	h=flip(tabjoueur,tabadversaire,c)
    tabjoueur⊻=h
    tabadversaire⊻=h
    tabjoueur=Bitboard.setindex(tabjoueur,true,c)
    moves=legalplay(tabadversaire,tabjoueur)
    Position(tabadversaire,tabjoueur,moves,-pos.player)
end


function isOver(pos)
	# if num_bit(pos.legalplay)!=0 || num_bit(legalplay(pos.bopponent,pos.bplayer))!=0
    #     return false,Int8(0)
    # else
        test=Int8(num_bit(pos.bplayer)-num_bit(pos.bopponent))
		# if test>0
        # 	return true,pos.player
		# elseif test==0
		# 	return true,Int8(0)
		# else
		# 	return true,-pos.player
		# end
		return num_bit(pos.legalplay)==0 && num_bit(legalplay(pos.bopponent,pos.bplayer))==0,sign(test)*pos.player

        # if test!=0
        #     return true,pos.player*test
        # elseif test==0
        #     return true,Int8(0)
        # else
        #     return true,-pos.player*test
        # end
    #end
end

function simul(prb)
	pos=Position()
    cpt=1
    pass=0
	while !(isOver(pos)[1])

        A=0f0
		for k in 1:maxActions
			if canPlay(pos,k)
				A+=1f0
			end
		end

        c=prb[cpt]*A
        move=1
        A=0f0
        for k in 1:maxActions
			if canPlay(pos,k)
				A+=1f0
                move=k
                if A>=c
                    break
                end
			end
		end
        if move==37
            pass+=1
        end

		pos=play(pos,move)
        cpt+=1

	end

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

function test(prb)
    RevSix.simul(prb)
    return
end
