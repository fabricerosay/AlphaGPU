module mcts_gpu
	export mcts,duelnetwork
using CUDA

using Flux

using Distributions
const  WIDTH = 7
const  HEIGHT = 6

struct Position
	mask::UInt64
	current_position::UInt64
	player::Int8
end

Position()=Position(0,0,1)

function canPlay(pos,col)
	  return (pos.mask & top_mask(col)) == 0
end


function  play(pos,col)

	  current_position = pos.mask ⊻ pos.current_position
	  mask = (pos.mask + bottom_mask(col)) | pos.mask
	  return Position(mask,current_position,-pos.player)

end



function isWinningMove(pos,col)
	  position = pos.current_position;
	  position |= (pos.mask + bottom_mask(col)) & column_mask(col);
	  return alignment(position);
end

function isOver(pos)
	current_position = pos.mask ⊻ pos.current_position
    if alignment(current_position)
        return true,-pos.player
    else
		moves=0
		for col in 0:6
			if canPlay(pos,col)
				moves+=1
			end
		end
		if moves==0
        	return true,0
		else
        	return false,0
		end
    end
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

function getActionNumber(pos)
	moves=0
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
        if canPlay(game,j)
            cpt+=1
        end
    end
    return j
end

function ksimul()
    pos=Position()
    p=1
	r=[]
    while true
        n=getActionNumber(pos)
        if n==0
            p=0
            break
        end
        k=ceil(Int,n*rand())
        col=findIndex(pos,k)
		push!(r,pos)
        if isWinningMove(pos,col)
            break
        else
            pos=play(pos,col)
        end
        p=-p
    end
    return r
end

function kmany_simul(vnodes)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    if index<=length(vnodes)
        @inbounds node=vnodes[index]
        @inbounds vnodes[index] = gpuNode(node.index,node.parent,node.actionFromParent,play(node.state,1),true)
    end
    return
end




function ktest(vnodes)
    numblocks = ceil(Int, length(vnodes)/256)
    @cuda threads=256 blocks=numblocks kmany_simul(vnodes)
    synchronize()
    return
end

function test(vnodes)


    many_simul(vnodes)
    return
end



struct gpuNode
    index::Int
    parent::Int
    actionFromParent::Int
    state::Position
    expanded::Bool
end



function create_nodes_stats(N,visits)
    (prior= zeros(Float32,(N,visits,7)),q=zeros(Float32,(N,visits,7)),visits=zeros(Float32,(N,visits,7)),child=zeros(Int,(N,visits,7)))
end

function create_cunodes_stats(N,visits)
    (prior=CuArray(zeros(Float32,(N,visits,7))),policy=CuArray(zeros(Float32,(N,visits,7))),q=CuArray(zeros(Float32,(N,visits,7))),visits=CuArray(zeros(Float32,(N,visits,7))),child=CuArray(zeros(Int,(N,visits,7))))
end

function create_roots(positions::Vector{Position},visits)
	nodes=Array{gpuNode}(undef,(length(positions),visits))
    for k in 1:length(positions)
		for j in 1:visits
			nodes[k,j]=gpuNode(j,0,0,positions[k],false)
		end
	end
	return nodes
end

function create_roots(N::Int,visits)
    nodes=Array{gpuNode}(undef,(N,visits))
	for k in 1:N
		for j in 1:visits
			nodes[k,j]=gpuNode(j,0,0,Position(),false)
		end
	end
    return nodes
end


@inbounds function kdescendTree!(leaves,vnodes,vnodesStats,newindex,prob)
     index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
     stride = blockDim().x * gridDim().x
    for i = index:stride:length(leaves)
    #for i in 1:length(leaves)
        current=vnodes[i,1]
		cpt=1
        while current.expanded
            nindex=current.index
            bestmove=-1
			p=0
			lastmove=1
			for l=1:7
                p+=vnodesStats.policy[i,nindex,l]
				if vnodesStats.prior[i,nindex,l]>0
                	bestmove=l
				end
                if p>=prob[i,cpt]
					break
                end

            end

            if vnodesStats.child[i,nindex,bestmove]==0
				newindex[i]+=1
                vnodesStats.child[i,nindex,bestmove]=newindex[i]
                vnodes[i,newindex[i]]=gpuNode(newindex[i],nindex,bestmove,play(current.state,bestmove-1),false)
            end
            current=vnodes[i,vnodesStats.child[i,nindex,bestmove]]
			cpt+=1
        end
        leaves[i]=current

    end
    return
end

function decoder(batch,leaf)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
	for i = index:stride:length(leaf)
		pos=leaf[i].state.current_position
		mask=leaf[i].state.mask
		for j in 0:48

			batch[j+1,i]=(pos>>j)&UInt(1)
			batch[j+49+1,i]=((pos⊻mask)>>j)&UInt(1)

		end
	end
	return
end


function decoder(position::Position)
	answer=falses(98)

	pos=position.current_position
	mask=position.mask
	for j in 0:48

		answer[j+1]=(pos>>j)&UInt(1)
		answer[j+50]=((pos⊻mask)>>j)&UInt(1)

	end

	return answer
end

function evaluate(batch,leaf,actor)
    numblocks = ceil(Int, length(leaf)/256)
    @cuda threads=256 blocks=numblocks decoder(batch,leaf)
    synchronize()
end

function expand(leaf,vnodes,vnodesStat,prior,dir)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    for i = index:stride:length(leaf)
        nindex=leaf[i].index
		f,r=isOver(leaf[i].state)
		vnodes[i,nindex]=gpuNode(nindex,leaf[i].parent,leaf[i].actionFromParent,leaf[i].state,!f)
		if !f
			if leaf[i].parent==0
				normalize=0
				for j in 1:7
					if canPlay(leaf[i].state,j-1)
						vnodesStat.prior[i,nindex,j]=0.75*prior[j,i]+0.25*dir[j,i]
						normalize+=vnodesStat.prior[i,nindex,j]
					end
				end

				for j in 1:7
					vnodesStat.prior[i,nindex,j]/=normalize
				end

			else
				normalize=0
				for j in 1:7
					if canPlay(leaf[i].state,j-1)
						vnodesStat.prior[i,nindex,j]=prior[j,i]
						normalize+=vnodesStat.prior[i,nindex,j]
					end
				end

				for j in 1:7
					vnodesStat.prior[i,nindex,j]/=normalize
				end
			end
		end
		for k in 1:7
			vnodesStat.policy[i,nindex,k]=vnodesStat.prior[i,nindex,k]
		end
	end
	return
end



function backUp(leaf,vnodes,vnodesStats,v)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    for i = index:stride:length(leaf)
        nindex = leaf[i].parent
        move = leaf[i].actionFromParent
		f,r=isOver(leaf[i].state)
		if f
			value=leaf[i].state.player*r
		else
			value=v[1,i]
		end
        while nindex!=0
            vnodesStats.q[i,nindex,move]+=(1-value)/2
			vnodesStats.visits[i,nindex,move]+=1
			N=1
			A=0
			for k in 1:7
				N+=vnodesStats.visits[i,nindex,k]
				if vnodesStats.prior[i,nindex,k]>0
					A+=1
				end
			end
			if A!=0
				λ=2*√(N)/(A+N)

				α=0f0

				for k in 1:7
					q=vnodesStats.q[i,nindex,k]/(vnodesStats.visits[i,nindex,k]+1)
					α=max(α,q+λ*vnodesStats.prior[i,nindex,k])
				end
				err=Inf32
				newerr=Inf32
				for j in 1:100
					S=0f0
	        		g=0f0
					for k in 1:7
						q=vnodesStats.q[i,nindex,k]/(vnodesStats.visits[i,nindex,k]+1)
						top=λ*vnodesStats.prior[i,nindex,k]
						bot=(α-q)
						S += top/bot
			            g += -top/(bot*bot)
						vnodesStats.policy[i,nindex,k]=top/bot
					end

        			newerr = S - 1f0

        			if newerr<0.001f0 || newerr==err
            			break
			        else
            			α -= newerr/g
            			err= newerr
					end
				end
			end
            move = vnodes[i,nindex].actionFromParent
            nindex= vnodes[i,nindex].parent
            value=-value
        end
    end
    return
end

function advance(vnodes,vnodesStats,prob,finished,results,ind)
	index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    for i = index:stride:length(finished)
		N=0
		if finished[i]==0
			if ind<20
				move=-1

				n=0
				for k in 1:7
					n+=vnodesStats.policy[i,1,k]
					if n>=prob[i] && canPlay(vnodes[i,1].state,k-1)
						move=k
						break
					end
				end
			else
				best=0
				move=-1
				for k in 1:7
					if vnodesStats.policy[i,1,k]>best && canPlay(vnodes[i,1].state,k-1)
						move=k
						best=vnodesStats.policy[i,1,k]
					end
				end
			end
			if move==-1
				@cuprint("bug")
				@cuprintln(prob[i])
				@cuprintln(n)
			end
			vnodes[i,1]=gpuNode(1,0,0,play(vnodes[i,1].state,move-1),false)

			f,r=isOver(vnodes[i,1].state)
			if f
				finished[i]=1
				results[i]=r
			end
		end
	end
	return
end

function push_results(states,vnodesStats,finalstates,policy,finished,ind,visits)
	index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
	for i = index:stride:length(finished)
		if finished[i]==0
			for k in 1:98
				finalstates[i,ind,k]=states[k,i]
			end
			for k in 1:7

				policy[i,ind,k]=vnodesStats.policy[i,1,k]
			end
		end
	end
	return
end

init(N,visits)=(vnodes=cu(create_roots(N,visits)),
vnodesStats=create_cunodes_stats(N,visits),
leaf=cu((Vector{gpuNode}(undef,N))),
batch=cu(zeros(98,N)),
newindex=cu(1*ones(Int,N)),finished=cu(zeros(N)),transferred=cu(falses(N)))

function mcts_single(positions,actor,visits)
	vnodes,vnodesStats,leaf,batch,newindex,finished,transferred=init(length(positions),visits)
	vnodes=cu(create_roots(positions,visits))
	a=1
	for k in 1:visits
		prob=cu(rand(length(positions),42))
        numblocks = ceil(Int, length(leaf)/256)
        @cuda threads=256 blocks=numblocks kdescendTree!(leaf,vnodes,vnodesStats,newindex,prob)
        synchronize()
        @cuda threads=256 blocks=numblocks decoder(batch,leaf)
        synchronize()

		prior,v=actor(batch)
		softmax!(prior)
        dir=cu(zeros(7,length(leaf)))
        @cuda threads=256 blocks=numblocks expand(leaf,vnodes,vnodesStats,prior,dir)
        synchronize()
        @cuda threads=256 blocks=numblocks backUp(leaf,vnodes,vnodesStats,v)
        synchronize()
    end
	Array(vnodesStats.policy[:,1,:])
end


function mcts(actor,visits,ngames)
	vnodes,vnodesStats,leaf,batch,newindex,finished,transferred=init(ngames,visits)
	states=similar(batch)
	finalstates=CuArray(zeros(ngames,42,98))
	policy=CuArray(zeros(ngames,42,7))
	results=CuArray(zeros(ngames))

	numblocks = ceil(Int, length(leaf)/256)
	ind=1

	while sum(finished)<ngames

		println(ind)
	    for k in 1:visits-1
			prob=cu(rand(ngames,42))
	        @cuda threads=256 blocks=numblocks kdescendTree!(leaf,vnodes,vnodesStats,newindex,prob)
	        synchronize()
	        @cuda threads=256 blocks=numblocks decoder(batch,leaf)
	        synchronize()


			prior,v=actor(batch)
			prior=prior.*0.8f0
			softmax!(prior)
	        dir=cu(rand(Dirichlet(7,1),length(leaf)))
	        @cuda threads=256 blocks=numblocks expand(leaf,vnodes,vnodesStats,prior,dir)
	        synchronize()
	        @cuda threads=256 blocks=numblocks backUp(leaf,vnodes,vnodesStats,v)
	        synchronize()
	    end
		states=cu(zeros(98,ngames))
		@cuda threads=256 blocks=numblocks decoder(states,vnodes[:,1])
		synchronize()
		@cuda threads=256 blocks=numblocks push_results(states,vnodesStats,finalstates,policy,finished,ind,visits)
		synchronize()
		prob=cu(rand(ngames))
		@cuda threads=256 blocks=numblocks advance(vnodes,vnodesStats,prob,finished,results,ind)
		synchronize()
		vnodesStats=create_cunodes_stats(ngames,visits)
		newindex=cu(1*ones(Int,ngames))
		ind=min(42,ind+1)
		println("finished games",sum(finished))

    end
	fs,pol,res=Array(finalstates),Array(policy),Array(results)
	r=[]
	for i in 1:ngames
		rtemp=[]
		for k in 1:42
			if sum(pol[i,k,:])!=0
				push!(r,(fs[i,k,:],(pol[i,k,:],(2(k%2)-1)*res[i])))
			end
		end
	end
	v=sum(res.==1)
	d=sum(res.==-1)
	n=ngames-(v+d)
	println("victoires,nul,défaites", [v,n,d])
	r
end

function mcts(actor1,actor2,visits,ngames)
	vnodes,vnodesStats,leaf,batch,newindex,finished,transferred=init(ngames,visits)
	results=CuArray(zeros(ngames))
	moves=cu(zeros(Int,(ngames,42)))
	numblocks = ceil(Int, length(leaf)/256)
	ind=1

	while sum(finished)<ngames

		if ind%2==1
			actor=actor1
		else
			actor=actor2
		end
	    for k in 1:visits-1
			prob=cu(rand(ngames,42))
	        @cuda threads=256 blocks=numblocks kdescendTree!(leaf,vnodes,vnodesStats,newindex,prob)
	        synchronize()
	        @cuda threads=256 blocks=numblocks decoder(batch,leaf)
	        synchronize()

			prior,v=actor(batch)
			softmax!(prior)
	        dir=cu(zeros(7,length(leaf)))
	        @cuda threads=256 blocks=numblocks expand(leaf,vnodes,vnodesStats,prior,dir)
	        synchronize()
	        @cuda threads=256 blocks=numblocks backUp(leaf,vnodes,vnodesStats,v)
	        synchronize()
	    end
		states=cu(zeros(98,ngames))
		@cuda threads=256 blocks=numblocks decoder(states,vnodes[:,1])
		synchronize()
		prob=cu(rand(ngames))
		@cuda threads=256 blocks=numblocks advance(vnodes,vnodesStats,prob,finished,results,ind)
		synchronize()
		vnodesStats=create_cunodes_stats(ngames,visits)
		newindex=cu(1*ones(Int,ngames))
		ind=min(42,ind+1)
    end
	r=Array(results)
	v=sum(r.==1)
	d=sum(r.==-1)
	n=ngames-(v+d)
	[v,n,d]
	
end

function duelnetwork(actor1,actor2,visits,ngames)
	hngames=div(ngames,2)
	println("net1 commence:")



	v1,n1,d1=mcts(actor1,actor2,visits,hngames)
	println("v:$v1 n:$n1 d:$d1")
	println("net2 commence:")
	d2,n2,v2=mcts(actor2,actor1,visits,hngames)
	println("v:$v2 n:$n2 d:$d2")
	return v1+v2,n1+n2,d1+d2
end
end
using .mcts_gpu
