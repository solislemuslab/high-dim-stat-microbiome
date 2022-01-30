# high-dim-stat-microbiome

The current optimization is using REML(Restricted MLE). The idea follows Theorem 17.4b in book [Linear Models in Statistics](http://www.utstat.toronto.edu/~brunner/books/LinearModelsInStatistics.pdf). 



The optimization in [MixedEffectModels.jl](https://github.com/JuliaStats/MixedModels.jl) is using a more general and computation easier (need verify) optimization [method](https://juliastats.org/MixedModels.jl/stable/optimization/#The-probability-model).



We have the same estimation result as in MixedEffectModels in real data. We will update out code into more general one as needed in the future.
