module HighDimMM

using LinearAlgebra: AbstractMatrix
using StatsModels
using LinearAlgebra
using DataFrames

##==============##==============##==============##==============##==============##==============##==============##==============##==============##==============##==============
# M
##==============##==============##==============##==============##==============##==============##==============##==============##==============##==============##==============

mutable struct highDimMat{T, S<:AbstractMatrix}
    M::S
end 


# constructor for highDimMat M,
function highDimMat(M::AbstractMatrix{T}) where {T}
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
mutable struct XMat{T, S<:AbstractMatrix}
    X::S
end 

# constructor for XMat X,
function XMat(X::AbstractMatrix{T}) where {T}
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
"""
function *(A::highDimMat{T}, B::XMat{T}) where{T}
    A.M*B.X
end
"""
#*(A::highDimMat{T, AbstractMatrix{T}}, B::XMat{T, AbstractMatrix{T}}) where{T} = A.M*B.X

*(A::highDimMat{Int64, Matrix{Int64}}, B::XMat{Int64, Matrix{Int64}}) where{T} = A.M*B.X

##==============##==============##==============##==============##==============##==============##==============##==============##==============##==============##==============
# Z
##==============##==============##==============##==============##==============##==============##==============##==============##==============##==============##==============
abstract type AbstractReMat{T} <: AbstractMatrix{T} end

mutable struct ReMat{T,S} <: AbstractReMat{T}
    trm # the grouping factor as a `StatsModels.CategoricalTerm`
    Z::Matrix{T}
end

LinearAlgebra.rank(A::ReMat) = rank(A.Z)

isfullrank(A::ReMat) = rank(A) == size(A.Z,2)

Base.getindex(A::ReMat, i::Int, j::Int) = getindex(A.X, i, j)

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
struct highDimMixedModel{T<:AbstractFloat} 
    formula::FormulaTerm
    M::highDimMat{T}
    X::XMat{T}
    Z::ReMat{T}
    y::Vector{T}
    optsum::OptSummary{T}
end

function highDimMixedModel(
    f::FormulaTerm
    df::DataFrame
    
)























end