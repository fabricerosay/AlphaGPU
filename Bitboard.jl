module Bitboard
import Base.getindex,Base.setindex,Base.length,Base.size,Base.:<<,Base.:>>>,Base.setindex!,Base.:~,Base.&,Base.:⊻,Base.:|,Base.copy
export bitboard,getindex,length,size,setindex!,down,left,right,num_bit,affiche

struct bitboard{N}
    chunks::NTuple{3,UInt64}
    len::Int
    dims::NTuple{N,Int}
end

length(bb::bitboard) = bb.len
size(bb::bitboard) = bb.dims

function bitboard{N}(dims::Vararg{Int,N}) where N
    n = 1
    i = 1
    for d in dims
        d >= 0 || throw(ArgumentError("dimension size must be ≥ 0, got $d for dimension $i"))
        n *= d
        i += 1
    end
    n<=192 || throw(ArgumentError("length must be inferior to 192 got $n"))
    b = bitboard((UInt64(0),UInt64(0),UInt64(0)), n,dims)
    return b
end

const _msk64 = ~UInt64(0)
@inline _div64(l) = l >> 6
@inline _mod64(l) = l & 63
@inline _blsr(x)= x & (x-1) #zeros the last set bit. Has native instruction on many archs. needed in multidimensional.jl
@inline _msk_end(l::Int) = _msk64 >>> _mod64(-l)

@inline function _msk(bb::bitboard)
    if length(bb)<=64
        return (_msk_end(length(bb)),UInt64(0),UInt64(0))
    elseif length(bb)<=128
        return (_msk64,_msk_end(length(bb)),UInt64(0))
    else
        return (_msk64,_msk64,_msk_end(length(bb)))
    end
end

num_bit_chunks(n::UInt64) = _div64(n+63)

@inline get_chunks_id(i::Int) = _div64(i-1)+1, _mod64(i-1)

@inline function getindex(bb::bitboard,i::Int)
    i1,i2=get_chunks_id(i)
    c=bb.chunks[i1]
    r = c & (UInt64(1)<<i2)!=0
    return r
end

@inline function getindex(bb::bitboard,i1::Int,i2::Int)
    i=bb.dims[1]*(i2-1)+i1
    return bb[i]
end


@inline function setindex(bb::bitboard, x::Bool,i::Int)
    i1,i2=get_chunks_id(i)
    u = UInt64(1) << i2
    c = bb.chunks[i1]
    newc=ifelse(x, c | u, c & ~u)
    #GC.@preserve bb unsafe_store!(Base.unsafe_convert(Ptr{UInt64}, pointer_from_objref(bb)), newc, i1)
    if i1==1
        return typeof(bb)((newc,bb.chunks[2],bb.chunks[3]),bb.len,bb.dims)
    elseif i1==2
        return typeof(bb)((bb.chunks[1],newc,bb.chunks[3]),bb.len,bb.dims)
    else
        return typeof(bb)((bb.chunks[1],bb.chunks[2],newc),bb.len,bb.dims)
    end

end

@inline function setindex(bb::bitboard,x,i1::Int,i2::Int)
    i=bb.dims[1]*(i2-1)+i1
    setindex!(bb,x,i)
end

function copy(bb::bitboard)
    return typeof(bb)(bb.chunks,bb.len,bb.dims)
end

function <<(bb::bitboard,n)
    i1=_div64(n)
    i2=_mod64(n)
    x,y,z=bb.chunks
    if 1<=i1<2
        z=y
        y=x
        x=UInt64(0)

    elseif i1>=2
        z=x
        x=UInt64(0)
        y=x
    end
    newx=x<<n
    headx=x>>>(64-i2)
    heady=y>>>(64-i2)
    newy=(y<<i2) | headx
    newz=(z<<i2) | heady
    mx,my,mz=_msk(bb)
    chunks=(newx&mx,newy&my,newz&mz)
    return typeof(bb)(chunks,bb.len,bb.dims)
end


function >>>(bb::bitboard,n)
    i1=_div64(n)
    i2=_mod64(n)
    x,y,z=bb.chunks
    if 1<=i1<2
        y=z
        z=0

        x=y

    elseif i1>=2
        x=z
        z=UInt64(0)
        y=z
    end
    newz=z>>>n
    headz=z<<(64-i2)
    heady=y<<(64-i2)
    newy=(y>>>i2) | headz
    newx=(x>>>i2) | heady
    mx,my,mz=_msk(bb)
    chunks=(newx&mx,newy&my,newz&mz)
    return typeof(bb)(chunks,bb.len,bb.dims)
end

function right(bb::bitboard)
    n=size(bb)[1]
    return bb<<n
end


function left(bb::bitboard)
    n=size(bb)[1]
    return bb>>>n
end

function down(bb::bitboard)
    dbb=bb<<1
    x,y,z=dbb.chunks
    for i in 1:size(bb)[1]:length(bb)
        i1,i2=get_chunks_id(i)
        if i1==1
            x&=~(UInt64(1)<<i2)
        elseif i1==2
            y&=~(UInt64(1)<<i2)
        else
            z&=~(UInt64(1)<<i2)
        end
    end
    return typeof(bb)((x,y,z),bb.len,bb.dims)
end

function num_bit(bb::bitboard)
    L=count_ones(bb.chunks[1])+count_ones(bb.chunks[2])+count_ones(bb.chunks[3])
    return L
end

function ~(bb::bitboard)
    x,y,z=bb.chunks
    mx,my,mz=_msk(bb)
    chunks=((~x)&mx,(~y)&my,(~z)&mz)
    return typeof(bb)(chunks,bb.len,bb.dims)
end

function Base.:&(bb1::bitboard,bb2::bitboard)
    x1,y1,z1=bb1.chunks
    x2,y2,z2=bb2.chunks
    return typeof(bb1)((x1&x2,y1&y2,z1&z2),bb1.len,bb1.dims)
end

function |(bb1::bitboard,bb2::bitboard)
    x1,y1,z1=bb1.chunks
    x2,y2,z2=bb2.chunks
    return typeof(bb1)((x1|x2,y1|y2,z1|z2),bb1.len,bb1.dims)
end
#
function ⊻(bb1::bitboard,bb2::bitboard)
    x1,y1,z1=bb1.chunks
    x2,y2,z2=bb2.chunks
    return typeof(bb1)((x1⊻x2,y1⊻y2,z1⊻z2),bb1.len,bb1.dims)
end


function affiche(bb::bitboard)
    b=zeros(Int64,size(bb))
    for x in 1:size(bb)[1], y in 1:size(bb)[2]
        b[x,y]=bb[x,y]
    end
    display(b)
end

end
