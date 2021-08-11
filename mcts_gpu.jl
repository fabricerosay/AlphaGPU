
module mcts_gpu

export mcts,duelnetwork

using CUDA
using Flux
using StatsBase
using Distributions
using StaticArrays
using Base.Threads
using ..Game
using Random:rand!
struct gpuNode
    parent::Int
    actionFromParent::Int
    state::Position
    expanded::Bool
    prior::Array
end


function unified_array(prealloc_array::Array{T,N}) where {T,N}
    buf = Mem.alloc(Mem.UnifiedBuffer, sizeof(prealloc_array))
    cpu_array = unsafe_wrap(Array, convert(Ptr{T}, buf), size(prealloc_array))
    gpu_array = unsafe_wrap(CuArray, convert(CuPtr{T}, buf), size(prealloc_array))
    return cpu_array, gpu_array, buf
end

function create_nodes_stats(L,visits)

    (prior= zeros(Float32,(L,visits,maxActions)),q=zeros(Float32,(L,visits,maxActions)),visits=zeros(Float32,(L,visits,maxActions)),child=zeros(Int,(L,visits,maxActions)))
end

function create_cunodes_stats(L,visits)
    (prior=CUDA.zeros(Float32,(L,visits,maxActions)),policy=CUDA.zeros(Float32,(L,visits,maxActions)),
    q=CUDA.zeros(Float32,(L,visits,maxActions)),visits=CUDA.zeros(Float32,(L,visits,maxActions)),
    Achild=CUDA.zeros(Int,(L,visits,maxActions)),childID=CUDA.zeros(Int,(L,visits,visits)),childnbr=CUDA.zeros(Int,(L,visits)),policy_final=CUDA.zeros(Float32,(L,maxActions)),batch=CUDA.zeros(Float32,(2*VectorizedState,L)))
end


function create_roots(positions::Vector{Position},visits)
    L=length(positions)
    state=Array{Position,2}(undef,(L,visits))
    for k in 1:L
		state[k,1]=positions[k]
	end
	nodes=(parent=CUDA.zeros(Int,(L,visits)),
           actionFromParent=CUDA.zeros(Int,(L,visits)),
           state=CuArray(state),
           expanded=CUDA.zeros(Int8,(L,visits)),uptodate=CUDA.fill(Int8(1),(L,visits)))
	return nodes
end

function create_roots(L::Int,visits)

    state=Array{Position,2}(undef,(L,visits))
    for k in 1:L
		state[k,1]=Position()
	end
	nodes=(parent=CUDA.zeros(Int,(L,visits)),
           actionFromParent=CUDA.zeros(Int,(L,visits)),
           state=CuArray(state),
           expanded=CUDA.zeros(Int8,(L,visits)),uptodate=CUDA.fill(Int8(1),(L,visits)))
	return nodes
end

function newton2(p,q,λ)
	α=0f0
	for k in 1:maxActions
		gap=max(λ*p[k],1f-4)
		α=max(α,q[k]+gap)
	end
	err=Inf32
	newerr=Inf32
	for j in 1:100
		S=0f0
		g=0f0
		for k in 1:maxActions
			top=λ*p[k]
			bot=(α-q[k])
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


@inbounds function kdescendTree!(leaves,vnodes,vnodesStats,newindex,prob,cpuct,L)

	 index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
     stride = blockDim().x * gridDim().x

    for i = index:stride:L
		#p=@cuStaticSharedMem(Float32,maxActions)#,maxActions*32*(2*(threadIdx().x-1)+1))
        #q=@cuStaticSharedMem(Float32,maxActions)#,maxActions*32*(2*(threadIdx().x-1)+1))
        nindex=1
		cpt=1
        while vnodes.expanded[i,nindex]==1
            bestmove=-1
			pr=0
			lastmove=1
            if vnodes.uptodate[i,nindex]!=1
            #if nindex==1
			    A=0f0
                n=1f0
				prior_rem=0f0
				childnbr=vnodesStats.childnbr[i,nindex]
                for k in 1:maxActions
                    n+=vnodesStats.visits[i,nindex,k]
					if vnodesStats.Achild[i,nindex,k]==0
						prior_rem+=vnodesStats.prior[i,nindex,k]
					end
                    #q[k]=vnodesStats.q[i,nindex,k]
                    #p[k]=vnodesStats.prior[i,nindex,k]

                    if vnodesStats.prior[i,nindex,k]>0
                        A+=1f0
                    end
                end
                λ=cpuct*sqrt(n)/(A+n)
                α=0f0
				prior_rem*=λ
                for k in 1:maxActions
                    gap=max(λ*vnodesStats.prior[i,nindex,k],1f-4)
                    α=max(α,vnodesStats.q[i,nindex,k]+gap)
                end
                err=Inf32
                newerr=Inf32
                for j in 1:100
                    S=prior_rem/α
                    g=-prior_rem/(α*α)
                    for k in 1:childnbr
						CID=vnodesStats.childID[i,nindex,k]
						action=vnodes.actionFromParent[i,CID]
                        top=λ*vnodesStats.prior[i,nindex,action]
                        bot=(α-vnodesStats.q[i,nindex,action])
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


                for k=1:maxActions
                    #p[k]=λ*p[k]/(α-q[k])

                    vnodesStats.policy[i,nindex,k]=λ*vnodesStats.prior[i,nindex,k]/(α-vnodesStats.q[i,nindex,k])#p[k]
                end
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

            if vnodesStats.Achild[i,nindex,bestmove]==0
				newindex[i]+=1
				vnodesStats.childnbr[i,nindex]+=1
				vnodesStats.childID[i,nindex,vnodesStats.childnbr[i,nindex]]=newindex[i]
                vnodesStats.Achild[i,nindex,bestmove]=vnodesStats.childnbr[i,nindex]
                vnodes.parent[i,newindex[i]]=nindex
                vnodes.actionFromParent[i,newindex[i]]=bestmove
                vnodes.state[i,newindex[i]]=play(vnodes.state[i,nindex],bestmove)
            end
            nindex=vnodesStats.childID[i,nindex,vnodesStats.Achild[i,nindex,bestmove]]
			cpt+=1
        end
        leaves[i]=nindex

    end
    return
end


function decoder(batch,vnodes,leaf,L)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
	for i = index:stride:L
		bplayer=vnodes.state[i,leaf[i]].bplayer
		bopponent=vnodes.state[i,leaf[i]].bopponent
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

function decoder_roots(batch,vnodes,L)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
	for i = index:stride:L
		bplayer=vnodes.state[i,1].bplayer
		bopponent=vnodes.state[i,1].bopponent
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



function expand(leaf,vnodes,vnodesStat,prior,noiseinit,training,L)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    for i = index:stride:L
        nindex=leaf[i]
		f,r=isOver(vnodes.state[i,nindex])
		vnodes.expanded[i,nindex]=Int8(1)-f

        if !f
			if nindex==1
				normalize=0

                A=0f0
				for j in 1:maxActions
					if canPlay(vnodes.state[i,nindex],j)
						vnodesStat.prior[i,nindex,j]=prior[j,i]
						normalize+=vnodesStat.prior[i,nindex,j]
                        A+=1f0
					end
				end
                if training
                    for j in 1:maxActions
                        if canPlay(vnodes.state[i,nindex],j)
    					    vnodesStat.prior[i,nindex,j]=0.75f0*vnodesStat.prior[i,nindex,j]/normalize+0.25f0/A
                        end
    				end
                else
                    for j in 1:maxActions
    					vnodesStat.prior[i,nindex,j]/=normalize
    				end
                end


			else
				normalize=0
				for j in 1:maxActions
					if canPlay(vnodes.state[i,nindex],j)
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



function backUp(leaf,vnodes,vnodesStats,v,L)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    for i = index:stride:L
        nindex = vnodes.parent[i,leaf[i]]
        move = vnodes.actionFromParent[i,leaf[i]]
		f,r=isOver(vnodes.state[i,leaf[i]])
		if f
			value=(1+vnodes.state[i,leaf[i]].player*r)/2
		else
			value=v[1,i]
		end
        while nindex!=0
            vnodesStats.q[i,nindex,move]=(vnodesStats.visits[i,nindex,move]*vnodesStats.q[i,nindex,move]+(1-value))/(vnodesStats.visits[i,nindex,move]+1)
			vnodesStats.visits[i,nindex,move]+=1
            vnodes.uptodate[i,nindex]=0
            move = vnodes.actionFromParent[i,nindex]
            nindex= vnodes.parent[i,nindex]
            value=1-value
        end
    end
    return
end

function adjust_policy(vnodesStats,vnodes,cpuct,L)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    for j = index:stride:64*L
        i=div(j-1,64)+1
        nindex=(j-1)%64+1
        if vnodes.uptodate[i,nindex]==false
            A=0f0
            n=1f0
            for k in 1:maxActions
                n+=vnodesStats.visits[i,nindex,k]
                #q[k]=vnodesStats.q[i,nindex,k]
                #p[k]=vnodesStats.prior[i,nindex,k]

                if vnodesStats.prior[i,nindex,k]>0
                    A+=1f0
                end
            end
            λ=cpuct*sqrt(n)/(A+n)
            α=0f0
            for k in 1:maxActions
                gap=max(λ*vnodesStats.prior[i,nindex,k],1f-4)
                α=max(α,vnodesStats.q[i,nindex,k]+gap)
            end
            err=Inf32
            newerr=Inf32
            for j in 1:100
                S=0f0
                g=0f0
                for k in 1:maxActions
                    top=λ*vnodesStats.prior[i,nindex,k]
                    bot=(α-vnodesStats.q[i,nindex,k])
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


            for k=1:maxActions
                #p[k]=λ*p[k]/(α-q[k])

                vnodesStats.policy[i,nindex,k]=λ*vnodesStats.prior[i,nindex,k]/(α-vnodesStats.q[i,nindex,k])#p[k]
            end
        end
    end
end

# function advance(vnodes,vnodesStats,prob,finished,results,ind)
# 	index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
#     stride = blockDim().x * gridDim().x
#     for i = index:stride:length(finished)
# 		N=0
# 		if finished[i]==0
# 			if ind<20
# 				move=-1
#
# 				n=0
# 				for k in 1:maxActions
# 					n+=vnodesStats.policy[i,1,k]
# 					if vnodesStats.policy[i,1,k]>0
# 						move=k
# 						if n>=prob[i]
# 							break
# 						end
# 					end
# 				end
# 			else
# 				best=0
# 				move=-1
# 				for k in 1:maxActions
# 					if vnodesStats.policy[i,1,k]>best
# 						move=k
# 						best=vnodesStats.policy[i,1,k]
# 					end
# 				end
# 			end
# 			if move==-1
# 				@cuprint("bug")
# 				@cuprintln(prob[i])
# 				@cuprintln(n)
# 			end
# 			vnodes.state[i,1]=play(vnodes.state[i,1],move)
#
# 			f,r=isOver(vnodes.state[i,1])
# 			if f
# 				finished[i]=1
# 				results[i]=r
# 			end
# 		end
# 	end
# 	return
# end
#
# function push_results(states,vnodesStats,finalstates,policy,values,finished,ind,visits)
# 	index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
#     stride = blockDim().x * gridDim().x
# 	for i = index:stride:length(finished)
# 		if finished[i]==0
# 			for k in 1:98
# 				finalstates[i,ind,k]=states[k,i]
# 			end
# 			q=0
# 			N=0
# 			for k in 1:7
# 				N+=vnodesStats.visits[i,1,k]
# 				q+=vnodesStats.q[i,1,k]
# 				policy[i,ind,k]=vnodesStats.policy[i,1,k]
# 			end
# 			values[i,ind]=q/N
# 		end
# 	end
# 	return
# end

function copy_pol(policy_final,policy,L)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    for i = index:stride:L
        for k in 1:maxActions
            policy_final[i,k]=policy[i,1,k]
        end
    end
return
end


function init(L,visits)
    vnodesStats=create_cunodes_stats(L,visits)
    (vnodes=create_roots(L,visits),
vnodesStats=vnodesStats,
leaf=CUDA.zeros(Int,L),
newindex=CUDA.fill(Int(1),L))
end

function init(positions::Vector{Position},visits)
    L=length(positions)
    vnodesStats=create_cunodes_stats(L,visits)
    return (vnodes=create_roots(positions,visits),
vnodesStats=vnodesStats,
leaf=CUDA.zeros(Int,L),
newindex=CUDA.fill(Int(1),L))
end

function reinit(positions,vnodes,L)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    for i = index:stride:L
        vnodes.state[i,1]=positions[i]
    end
    return
end

function re_init(positions,vnodes,L,nthreads,numblocks)
    @cuda threads=nthreads blocks=numblocks reinit(positions,vnodes,L)
    CUDA.synchronize()
    vnodes.expanded.=0
    vnodes.uptodate.=1
end


function mcts_single(actor,visits,nthreads,vnodes,vnodesStats,leaf,newindex,L;training=true,cpuct=2f0,noise=Float32(1/maxActions),conv=false)
    t7=time()
    numblocks = ceil(Int, L/nthreads)
    t0=time()
    vnodesStats.q.=0
    vnodesStats.Achild.=0
	vnodesStats.childID.=0
    vnodesStats.visits.=0
    vnodesStats.prior.=0
    vnodesStats.policy.=0
	vnodesStats.childnbr.=0
    newindex.=1
    t0=time()-t0
	t1=t2=t3=t4=t5=0
	ttot=time


    # prior,v=actor(view(vnodesStats.batch,:,1:L),training=false)
    # softmax!(prior)

	for k in 1:visits
		prob=CUDA.rand(L,maxLengthGame)
        synchronize()
		t=time()
        #println("descent")
        @cuda threads=nthreads blocks=numblocks kdescendTree!(leaf,vnodes,vnodesStats,newindex,prob, cpuct,L)
        synchronize()
		t1+=time()-t
		t=time()
		CUDA.unsafe_free!(prob)
         @cuda threads=nthreads blocks=numblocks decoder(vnodesStats.batch,vnodes,leaf,L)
        synchronize()
		t2+=time()-t

		t=time()

		prior,v=actor(view(vnodesStats.batch,:,1:L))


		 softmax!(prior)
        synchronize()

		t5+=time()-t

        t=time()

         @cuda threads=nthreads blocks=numblocks expand(leaf,vnodes,vnodesStats,prior,noise,training,L)
        synchronize()
		t3+=time()-t
		CUDA.unsafe_free!(prior)
        t=time()

         @cuda threads=nthreads blocks=numblocks backUp(leaf,vnodes,vnodesStats,v,L)
        synchronize()
		t4+=time()-t

		CUDA.unsafe_free!(v)

        #ccall(:malloc_trim,Cvoid,(Cint,),0)
        # @cuda threads=nthreads blocks=64*numblocks adjust_policy(vnodesStats,vnodes,cpuct,L)
        # synchronize()
    end
    t6=time()
	 @cuda threads=nthreads blocks=numblocks decoder_roots(vnodesStats.batch,vnodes,L)
	synchronize()
    @cuda threads=nthreads blocks=numblocks copy_pol(vnodesStats.policy_final,vnodesStats.policy,L)
    synchronize()
    t6=time()-t6
    t7=time()-t7
    #println(t7)

    #CUDA.unsafe_free!(prior)
    #CUDA.unsafe_free!(v)
    # println(t0)
    # println(t1)
    # println(t2)
    # println(t3)
    # println(t4)
    # println(t5)
    return
    # println(t6)
    # println(t0+t1+t2+t3+t4+t5+t6)
	#Array(batch),Array(vnodesStats.policy[:,1,:])#,Array(sum(vnodesStats.q[:,1,:],dims=3)./visits)

end

function decode(pos)
    fstate=zeros(Int8,VectorizedState)
    for j in 1:VectorizedState
        if pos.bplayer[j]
            fstate[j]=pos.player
        else
            fstate[j]=-pos.player
        end
    end
    return fstate
end


function mcts(actor,visits,ngames,buffer::Main.PoolSample;θ=1,cpuct=2.0,noise=Float32(1/maxActions))
	ttot=time()
	positions=[Position() for k in 1:ngames]
    truevisits=visits
    (vnodes,vnodesStats,leaf,newindex)=init(positions,visits)
	rtemp=[Int[] for k in 1:ngames]
	r=[]
	round=0
	v=0
	d=0
	n=0
	GC.gc(true)
	CUDA.reclaim()
    L=ngames
    blunders=0
    tot_length=0
    record=true
	while !isempty(positions)
		t1=t=time()
        # if rand()<0.25
        #     record=true
        #     truevisits=4*visits
        # else
        #     record=true
        #     truevisits=visits
        # end
		mcts_single(actor,truevisits,256,vnodes,vnodesStats,leaf,newindex,L,cpuct=cpuct,noise=noise)

        policy,batch=Array(vnodesStats.policy_final),Array(vnodesStats.batch)
		t1=time()-t1
		#policy=[π[i,:] for i in 1:length(positions)]
		#batch=[batch[:,i] for i in 1:length(positions)]
		i=1
        finished=[]
        t2=time()
		for  i in 1:length(positions)
			index=Main.push_buffer(buffer,batch,policy,positions[i].player,i)
			push!(rtemp[i],index)
			pol=buffer.pool[index].policy
			if round<25
				lp=[c for c in 1:maxActions if pol[c]!=0]
				c=sample(lp,Weights(pol[lp]))

			else
				c=argmax(pol)[1]
			end
			buffer.pool[index].move[1]=c


			if !canPlay(positions[i],c)
				println("faute")
			    return (data=(positions[i],pol,c),valid=false)
			end
			positions[i]=play(positions[i],c)
			f,res=isOver(positions[i])
			if f
                fstate=decode(positions[i])
                push!(finished,i)
                tot_length+=round


				Main.update_buffer(buffer,rtemp[i],res,fstate)


				if res==1
					v+=1
				elseif res==0
					n+=1
				else
					d+=1
				end
			end
		end
        for (k,c) in enumerate(finished)
            deleteat!(rtemp,c-k+1)
            deleteat!(positions,c-k+1)
        end
			#policy=nothing
			#batch=nothing
			round+=1
            if L>0
            L=length(positions)
            numblocks = max(1,ceil(Int, L/1024))
            re_init(cu(positions),vnodes,L,1024,numblocks)
            end
			t2=time()-t2
            ttot_par=t2+t1
			finished=ngames-length(positions)
			println("round:$round, temps total=$ttot_par,finished=$finished")
			println("temps mcts=$t1, temps gestions=$t2")
            println("bluders number :$blunders")


    end
	#ccall(:malloc_trim,Cvoid,(Cint,),0)
	#GC.gc()
	ttot=time()-ttot
	println("victoires,nul,défaites", [v,n,d])
	println("temps total=$ttot")
    mean_length=tot_length/ngames
    println("longueur moyenne: $mean_length")
	(data=r,valid=true)
end

function mcts(actor1,actor2,visits,ngames;cpuct=2f0,noise=Float32(1/maxActions),conv=2)
	ttot=time()
    positions=[Position() for k in 1:ngames]
    (vnodes,vnodesStats,leaf,newindex)=init(length(positions),visits)
	round=0
	v=0
	d=0
	n=0
    L=ngames
	while !isempty(positions)
		t=time()
		if round%2==0
			actor=actor1
		else
			actor=actor2
		end
        finished=[]
        use_conv=(conv%2!=round%2)&(conv!=-1)
		mcts_single(actor,visits,256,vnodes,vnodesStats,leaf,newindex,L,training=false,noise=noise,cpuct=cpuct,conv=use_conv)
        policy=Array(vnodesStats.policy_final)
        i=1

		for  i in 1:length(positions)

			if round<15
				c=sample(1:maxActions,Weights(@view policy[i,:]))
			else
				c=argmax(@view policy[i,:])
			end
			#println("c=$c,player=",positions[i].player,"  ",argmax(π))
			if !canPlay(positions[i],c)
				println("faute")
			    return positions[i],policy[i,:]
			end
			positions[i]=play(positions[i],c)
			f,res=isOver(positions[i])
			if f

                push!(finished,i)
				if res==1
					v+=1
				elseif res==0
					n+=1
				else
					d+=1
				end

			end
		end
        for (k,c) in enumerate(finished)
            deleteat!(positions,c-k+1)
        end
		policy=nothing
			round+=1
            if L>0
            L=length(positions)
            numblocks = max(1,ceil(Int, L/256))
            re_init(cu(positions),vnodes,L,256,numblocks)
            end
			t=time()-t
			finished=ngames-length(positions)
			#println("round:$round, temps=$t,finished=$finished")

    end
	ccall(:malloc_trim,Cvoid,(Cint,),0)
	GC.gc()
	ttot=time()-ttot

	println("temps total=$ttot")
	[v,n,d]
end

function duelnetwork(actor1,actor2,visits,ngames,conv=2)
	hngames=div(ngames,2)
	println("net1 commence:")



	v1,n1,d1=mcts(actor1,actor2,visits,hngames,conv=conv)
	println("v:$v1 n:$n1 d:$d1")
	println("net2 commence:")
    if conv==-1
        conv=4
    end
	d2,n2,v2=mcts(actor2,actor1,visits,hngames,conv=3-conv)
	println("v:$v2 n:$n2 d:$d2")
	return v1+v2,n1+n2,d1+d2
end
end
