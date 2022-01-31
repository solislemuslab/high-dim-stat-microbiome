include("bricks.jl")
using LinearAlgebra

A = HighDimMM.highDimMat([1 2 3 4;3 4 5 6])
B = HighDimMM.XMat([1 2 3;3 4 5; 0 0 1; 1 0 0])

HighDimMM.*(A,B)