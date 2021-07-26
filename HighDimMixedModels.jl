module HighDimMM

using Base: Float64
using DataFrames: Dict
#using LinearAlgebra: AbstractMatrix, include
using StatsModels
using LinearAlgebra
using DataFrames

import Base: *

abstract type MixedModel{T} <: StatsModels.RegressionModel end # model with fixed and random effects

include("bricks.jl")

export highDimMixedModel


end