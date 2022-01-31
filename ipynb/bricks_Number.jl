##==============##==============##==============##==============##==============##==============##==============##==============##==============##==============##==============
# M
##==============##==============##==============##==============##==============##==============##==============##==============##==============##==============##==============
mutable struct highDimMat{T, S<:AbstractVecOrMat}
    M::S
end 


# constructor for highDimMat M,
function highDimMat(M::AbstractVecOrMat{T}) where {T}
    if size(M,1) >= size(M,2)
        @warn "n >= p in high dimensional matrix"
    end
    highDimMat{T, typeof(M)}(M)
end


"""
# constructor for highDimMat M,
function highDimMat(M::AbstractMatrix)
    if size(M,1) >= size(M,2)
        @warn "n >= p in high dimensional matrix"
    end
    return highDimMat{eltype(M), typeof(M)}(M)
end
"""

Base.copyto!(A::highDimMat{T}, src::AbstractVecOrMat{T}) where {T} = copyto!(A.M, src)

Base.eltype(::highDimMat{T}) where {T} = T

Base.getindex(A::highDimMat{T}, i::Int, j::Int) where {T} = getindex(A.M, i, j)

Base.size(A::highDimMat{T}) where {T} = size(A.M)

Base.size(A::highDimMat{T}, i::Integer) where {T} = size(A.M, i)
##==============##==============##==============##==============##==============##==============##==============##==============##==============##==============##==============
# X
##==============##==============##==============##==============##==============##==============##==============##==============##==============##==============##==============
mutable struct XMat{T, S<:AbstractVecOrMat}
    X::S
end 

# constructor for XMat X,
function XMat(X::AbstractVecOrMat{T}) where {T}
    if rank(X) < size(X,2)
        @warn "fixed effect matrix is not of full rank"
    end

    if size(X,1) < size(X,2)
        @warn "n < p in covariate matrix X"
    end
    XMat{T, typeof(X)}(X)
end

Base.copyto!(A::XMat{T}, src::AbstractVecOrMat{T}) where {T} = copyto!(A.X, src)

Base.eltype(::XMat{T}) where {T} = T

LinearAlgebra.rank(A::XMat{T}) where {T} = rank(A.X)

isfullrank(A::XMat{T}) where {T} = rank(A) == size(A.X,2)

Base.getindex(A::XMat{T}, i::Int, j::Int) where {T} = getindex(A.X, i, j)

#Base.adjoint(A::XMat{T}) = Adjoint(A)

Base.size(A::XMat{T}) where {T} = size(A.X)

Base.size(A::XMat{T}, i::Integer) where {T} = size(A.X, i)

## define new infix binary operator?

function Base.:*(A::highDimMat{T}, B::XMat{T}) where{T}
    A.M*B.X
end


#*(A::highDimMat{T, AbstractMatrix{T}}, B::XMat{T, AbstractMatrix{T}}) where{T} = A.M*B.X

*(A::highDimMat{Int64, Matrix{Int64}}, B::XMat{Int64, Matrix{Int64}}) where{T} = A.M*B.X

##==============##==============##==============##==============##==============##==============##==============##==============##==============##==============##==============
# Z
##==============##==============##==============##==============##==============##==============##==============##==============##==============##==============##==============
abstract type AbstractReMat{T} <: AbstractMatrix{T} end

mutable struct ReMat{T,S} <: AbstractMatrix{T}
    #trm # the grouping factor as a `StatsModels.CategoricalTerm`   ##????
    Z::Matrix{T}
end

# constructor for XMat X,
function ReMat(Z::AbstractMatrix{T}) where {T}
    ReMat{T, typeof(Z)}(Z)
end



LinearAlgebra.rank(A::ReMat) = rank(A.Z)

isfullrank(A::ReMat) = rank(A) == size(A.Z,2)

Base.getindex(A::ReMat, i::Int, j::Int) = getindex(A.Z, i, j)

Base.size(A::ReMat) = size(A.Z)


##==============##==============##==============##==============##==============##==============##==============##==============##==============##==============##==============
# highDimMixedModel
##==============##==============##==============##==============##==============##==============##==============##==============##==============##==============##==============
"""
    highDimMixedModel
High dim mixed-effects model representation
## Fields
* `formula`: the formula for the model
* `optsum`: an [`OptSummary`](@ref) object
## Properties
* `θ` or `theta`: the covariance parameter vector used to form λ
* `β` or `beta`: the fixed-effects coefficient vector
* `λ` or `lambda`: a vector of lower triangular matrices repeated on the diagonal blocks of `Λ`
* `σ` or `sigma`: current value of the standard deviation of the per-observation noise
* `b`: random effects on the original scale, as a vector of matrices
* `u`: random effects on the orthogonal scale, as a vector of matrices
* `lowerbd`: lower bounds on the elements of θ
* `X`: the fixed-effects model matrix
* `y`: the response vector
"""
struct highDimMixedModel{T<:Number}  <: MixedModel{T}
    formula::FormulaTerm
    M::highDimMat{T}
    X::XMat{T}
    Z::ReMat{T}
    y::Vector{T}
    #optsum::OptSummary{T}
end


"""
    highDimMixedModel
dealing with the number of columns in M and X scenario, assume intercept alwalys exists (@formula: y ~ 1 + ... )
"""
function highDimMixedModel(
    f::FormulaTerm,
    df::DataFrame,
    contrasts::Dict{Symbol, UnionAll},
    numOfHDM::Int64,
    numOfXMat::Int64
) where{T}
    sch = schema(df,contrasts)
    form = apply_schema(f, sch)
    y, pred = modelcols(form, df);
    MXZ = pred[:,2:size(pred,2)]  ## get rid of intercept
    M = highDimMat(MXZ[:,1:numOfHDM])
    intercept = pred[:,1]
    X = XMat(hcat(reshape(intercept, size(intercept,1),1), MXZ[:, (numOfHDM + 1):(numOfHDM + numOfXMat)]))  ## concatenate intercept with X
    Z = ReMat(MXZ[:, (numOfHDM + numOfXMat + 1):size(MXZ,2)])
    
    #return highDimMixedModel{T<:AbstractFloat}(form, M, X, Z, y)
    return highDimMixedModel{<:Number}(form, M, X, Z, y)
end

""" not useful right now
private help function for select indices for remaining random effect matrix
    preTerms is Vector(1:length(form.rhs.terms))

_filt(x, preTerms) = !(x in preTerms)

preTerms = Vector(1:length(form.rhs.terms))

"""

function highDimMixedModel(
    f::FormulaTerm,
    df::DataFrame,
    contrasts::Dict{Symbol, UnionAll},
    idOfHDM::Union{Int,AbstractArray{Int64,1}}, # [1,2]
    idOfXMat::Union{Int,AbstractArray{Int64,1}},
    idOfReMat::Union{Int,AbstractArray{Int64,1}}
) where {T} 
    """
    for i in 1:size(df,2)
        if(eltype(df[:,i]) == Int64)
            df[!,i] = convert(Vector{Float64},df[:,i])
        end
    end
    """
    sch = schema(df, contrasts)
    form = apply_schema(f, sch)
    y, pred = modelcols(form, df);
    terms = form.rhs.terms
    M = highDimMat(modelmatrix(terms[idOfHDM],df))
    X = XMat(modelmatrix(terms[idOfXMat],df))

    #preTerms = Vector(1:length(form.rhs.terms))
    #idOfReMat = preTerms[preTerms .∉ Ref(vcat(idOfHDM, idOfXmat))]
    Z = ReMat(modelmatrix(terms[idOfReMat],df))

    return highDimMixedModel{<:Number}(form, M, X, Z, y)

end

##==============##==============##==============##==============##==============##==============##==============##==============##==============##==============##==============
# fit
##==============##==============##==============##==============##==============##==============##==============##==============##==============##==============##==============
"""
function fit(
    ::Type{highDimMixedModel},
    f::FormulaTerm,
    df::DataFrame;
    contrasts,
    progress,
    REML,
    idOfHDM::Union{Int,AbstractArray{Int64,1}}, # [1,2]
    idOfXMat::Union{Int,AbstractArray{Int64,1}},
    idOfReMat::Union{Int,AbstractArray{Int64,1}}
) return fit!(highDimMixedModel(f, df, contrasts, idOfHDM, idOfXMat, idOfReMat), progress, REML)
end
    
function fit(
    HMM::Type{highDimMixedModel},
    progress,
    REML
) return fit!(HMM, progress, REML)
end
"""


"""
private help function: objective
"""

function negLogLik(sigma::Vector, grad::Vector, K::AbstractVecOrMat, Z::AbstractMatrix, y::Vector)
    n = length(y)
    Sigma = sigma[1]*Z*transpose(Z) + sigma[2]*diagm(ones(n))
    negLog = -1/2*log(det(K*Sigma*K)) - 1/2*transpose(y)*transopse(K)*inv(K*Sigma*K)*K*y
    return negLog
end

function fit!(HMM::LinearMixedModel{T}; progress::Bool=true, REML::Bool=false) where {T}
    A = hcat(HMM.M.M, HMM.X.X)
    P = I - A*inv(transpose(A)*A)*transpose(A)
    u,s,v = svd(P)
    r = size(HMM.M,2) + size(HMM.X,2)  # simplify: assume fixed effect full rank
    C = transpose(u[:,1:r]) 
    K = C*P
    
    opt = Opt(:LN_COBYLA, 2)
    opt.min_objective = negLogLik
    opt.xtol_rel = 1e-5

    (optf,optx,ret) = optimize(opt, [1,1])


end