module mcts_gpu

export mcts,duelnetwork

using CUDA
using Flux
using StatsBase
using Distributions

using ..GoBang

struct gpuNode
    index::Int
    parent::Int
    actionFromParent::Int
    state::Position
    expanded::Bool
end



function create_nodes_stats(L,visits)
    (prior= zeros(Float32,(L,visits,maxActions)),q=zeros(Float32,(L,visits,maxActions)),visits=zeros(Float32,(L,visits,maxActions)),child=zeros(Int,(L,visits,maxActions)))
end

function create_cunodes_stats(L,visits)
    (prior=CuArray(zeros(Float32,(L,visits,maxActions))),policy=CuArray(zeros(Float32,(L,visits,maxActions))),q=CuArray(zeros(Float32,(L,visits,maxActions))),visits=CuArray(zeros(Float32,(L,visits,maxActions))),child=CuArray(zeros(Int,(L,visits,maxActions))))
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

function create_roots(L::Int,visits)
    nodes=Array{gpuNode}(undef,(L,visits))
	for k in 1:L
		for j in 1:visits
			nodes[k,j]=gpuNode(j,0,0,Position(),false)
		end
	end
    return nodes
end

function newton(p,λ)
	α=0f0
	for k in 1:maxActions
		gap=max(λ*p[2*k-1],1f-4)
		α=max(α,p[2*k]+gap)
	end
	err=Inf32
	newerr=Inf32
	for j in 1:100
		S=0f0
		g=0f0
		for k in 1:maxActions
			top=λ*p[2*k-1]
			bot=(α-p[2*k])
			S += top/bot
			g += -top/(bot*bot)
		end

		newerr = S - 1f0

		if newerr<0.001f0 || newerr==err
			break
		else
			α -= newerr/g
			err= newerr
		end
	end
	return α
end


@inbounds function kdescendTree!(leaves,vnodes,vnodesStats,newindex,prob)

	 index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
     stride = blockDim().x * gridDim().x

    for i = index:stride:length(leaves)
		p=@cuDynamicSharedMem(Float32,2*maxActions,maxActions*32*(2*(threadIdx().x-1)))

        current=vnodes[i,1]
		cpt=1
        while current.expanded
            nindex=current.index
            bestmove=-1
			pr=0
			lastmove=1

			A=0f0
			n=1f0
			for k in 1:maxActions
				n+=vnodesStats.visits[i,nindex,k]
				p[2*k]=vnodesStats.q[i,nindex,k]/(vnodesStats.visits[i,nindex,k]+1)
				p[2*k-1]=vnodesStats.prior[i,nindex,k]

				if vnodesStats.prior[i,nindex,k]>0
					A+=1f0
				end
			end
			λ=2*sqrt(n)/(A+n)
			α=newton(p,λ)

			for k=1:maxActions
				p[2*k-1]=λ*p[2*k-1]/(α-p[2*k])

				vnodesStats.policy[i,nindex,k]=p[2*k-1]
			end

			for k=1:maxActions
				Δ=vnodesStats.policy[i,nindex,k]
                pr+=Δ#p[2*k-1]
				if Δ>0
                	bestmove=k
				end
                if pr>=prob[i,cpt]
					break
                end

            end

            if vnodesStats.child[i,nindex,bestmove]==0
				newindex[i]+=1
                vnodesStats.child[i,nindex,bestmove]=newindex[i]
                vnodes[i,newindex[i]]=gpuNode(newindex[i],nindex,bestmove,play(current.state,bestmove),false)
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
		bplayer=leaf[i].state.bplayer
		bopponent=leaf[i].state.bopponent
		for j in 1:VectorizedState
			if bplayer[j]
				batch[j,i]=1
			else
				batch[j,i]=0
			end
			if bopponent[j]
				batch[j+VectorizedState,i]=1
			else
				batch[j+VectorizedState,i]=0
			end

		end
	end
	return
end

function decoder_positions(batch,positions)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
	for i = index:stride:length(positions)
		bplayer=positions[i].bplayer
		bopponent=positions[i].bopponent
		for j in 1:VectorizedState
			if bplayer[j]
				batch[j,i]=1
			else
				batch[j,i]=0
			end
			if bopponent[j]
				batch[j+VectorizedState,i]=1
			else
				batch[j+VectorizedState,i]=0
			end

		end
	end
	return
end


function decoder(position::Position)
	answer=falses(VectorizedState)

	pos=position.current_position
	mask=position.mask
	for j in 0:48

		answer[j+1]=(pos>>j)&UInt(1)
		answer[j+50]=((pos⊻mask)>>j)&UInt(1)

	end

	return answer
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
				for j in 1:maxActions
					if canPlay(leaf[i].state,j)
						vnodesStat.prior[i,nindex,j]=0.75*prior[j,i]+0.25*1/maxActions
						normalize+=vnodesStat.prior[i,nindex,j]
					end
				end

				for j in 1:maxActions
					vnodesStat.prior[i,nindex,j]/=normalize
				end

			else
				normalize=0
				for j in 1:maxActions
					if canPlay(leaf[i].state,j)
						vnodesStat.prior[i,nindex,j]=prior[j,i]
						normalize+=vnodesStat.prior[i,nindex,j]
					end
				end

				for j in 1:maxActions
					vnodesStat.prior[i,nindex,j]/=normalize
				end
			end
		end
		for k in 1:maxActions
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
            move = vnodes[i,nindex].actionFromParent
            nindex= vnodes[i,nindex].parent
            value=-value
        end
    end
    return
end



init(L,visits)=(vnodes=cu(create_roots(L,visits)),
vnodesStats=create_cunodes_stats(L,visits),
leaf=cu((Vector{gpuNode}(undef,L))),
batch=cu(zeros(2*VectorizedState,L)),
newindex=cu(1*ones(Int,L)),finished=cu(zeros(L)),transferred=cu(falses(L)))


function mcts_single(positions,actor,visits,nthreads;training=true)
	vnodes,vnodesStats,leaf,batch,newindex,finished,transferred=init(length(positions),visits)
	vnodes=cu(create_roots(positions,visits))
	numblocks = ceil(Int, length(leaf)/nthreads)
	t0=t1=t2=t3=t4=t5=0
	ttot=time
	dir=CUDA.zeros(Float32,(maxActions,length(leaf)))
	for k in 1:visits
		prob=cu(rand(length(positions),maxLengthGame))
		t=time()
        @cuda threads=nthreads blocks=numblocks shmem=maxActions*64*nthreads kdescendTree!(leaf,vnodes,vnodesStats,newindex,prob)
        synchronize()
		t1+=time()-t
		t=time()
        @cuda threads=nthreads blocks=numblocks decoder(batch,leaf)
        synchronize()
		t2+=time()-t
		t=time()
		prior,v=actor(batch)
		softmax!(prior)
		t5+=time()-t
		t=time()
		synchronize()
		t0+=time()-t
		t=time()
        @cuda threads=nthreads blocks=numblocks expand(leaf,vnodes,vnodesStats,prior,dir)
        synchronize()
		t3+=time()-t
		t=time()
        @cuda threads=nthreads blocks=numblocks backUp(leaf,vnodes,vnodesStats,v)
        synchronize()
		t4+=time()-t
    end
	@cuda threads=nthreads blocks=numblocks decoder_positions(batch,cu(positions))
	synchronize()


	Array(batch),Array(vnodesStats.policy[:,1,:])

end




function mcts(actor,visits,ngames;θ=1)
	ttot=time()
	positions=[Position() for k in 1:ngames]
	rtemp=[[] for k in 1:ngames]
	r=[]
	round=0
	v=0
	d=0
	n=0
	GC.gc(true)
	CUDA.reclaim()
	while !isempty(positions)
		t=time()
		batch,π=mcts_single(positions,actor,visits,8)
		t1=time()-t
		policy=[π[i,:] for i in 1:length(positions)]
		batch=[batch[:,i] for i in 1:length(positions)]
		i=1
		while i<length(positions)+1


			push!(rtemp[i],(batch[i],(policy[i],positions[i].player)))

			if round<15
				c=sample(1:maxActions,Weights(policy[i]))
			else
				c=argmax(policy[i])
			end

			if !canPlay(positions[i],c)
				println("faute")
			    return positions[i],policy[i,:]
			end
			positions[i]=play(positions[i],c)
			f,res=isOver(positions[i])
			if f
				for el in rtemp[i]
					x,(y,z)=el
					push!(r,(x,(y,z*res)))
				end
				if res==1
					v+=1
				elseif res==0
					n+=1
				else
					d+=1
				end
				deleteat!(rtemp,i)
				deleteat!(positions,i)
				deleteat!(policy,i)
				deleteat!(batch,i)
			else
				i+=1
			end
		end
			round+=1
			t=time()-t
			finished=ngames-length(positions)
			println("round:$round, temps total=$t,finished=$finished")
			t2=t-t1
			println("temps mcts=$t1, temps gestions=$t2")

    end
	ttot=time()-ttot
	println("victoires,nul,défaites", [v,n,d])
	println("temps total=$ttot")
	r
end

function mcts(actor1,actor2,visits,ngames)
	ttot=time()
	positions=[Position() for k in 1:ngames]
	round=0
	v=0
	d=0
	n=0
	while !isempty(positions)
		t=time()
		if round%2==0
			actor=actor1
		else
			actor=actor2
		end
		batch,π=mcts_single(copy(positions),actor,visits,8,training=false)
		policy=[π[i,:] for i in 1:length(positions)]
		i=1
		while i<length(positions)+1

			if round<15
				c=sample(1:maxActions,Weights(policy[i]))
			else
				c=argmax(policy[i])
			end
			#println("c=$c,player=",positions[i].player,"  ",argmax(π))
			if !canPlay(positions[i],c)
				println("faute")
			    return positions[i],policy[i,:]
			end
			positions[i]=play(positions[i],c)
			f,res=isOver(positions[i])
			if f
				if res==1
					v+=1
				elseif res==0
					n+=1
				else
					d+=1
				end
				deleteat!(positions,i)
				deleteat!(policy,i)
			else
				i+=1
			end
		end
			round+=1
			t=time()-t
			finished=ngames-length(positions)
			#println("round:$round, temps=$t,finished=$finished")

    end
	ttot=time()-ttot

	println("temps total=$ttot")
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
