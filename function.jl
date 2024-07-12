module function 

using LinearAlgebra
using SparseArrays
using Plots
using Base.Threads
using Distributed
using IterativeSolvers
using LinearRegression
using LinearMaps
using Arpack

#potentiel
V(x) = -20 * (exp(-30 * cos(pi * (x - 0.20))^2) + 2 * exp(-30 * cos(pi * (x + 0.25))^2))

function projection(P::AbstractMatrix{T}, X::AbstractMatrix{T}) where T #projection sur l'espace tangent à la matrice P
    n = size(P, 1)
    if isa(P, Symmetric{T})
        P = Matrix(P)
    end

    proj = P * X * (I - P) + (I - P) * X * P

    return proj
end

function construction(Nb) #construction du hamiltonien
   x = LinRange(0, 1, Nb)
    diag_V = V.(x)
    main_diag = (1 / dx^2) .+ diag_V
    off_diag = fill(-1 / (2 * dx^2), Nb - 1)

    h = spdiagm(0 => main_diag, 1 => off_diag, -1 => off_diag)
    h[1, Nb] = h[Nb, 1] = -1 / (2 * dx^2)
    return Matrix(h)
end

function phi(A)     #Matrice des vecteurs propres triés
    eigvals, eigvecs = eigen(A)
    sorted_indices = sortperm(eigvals)
    sorted_eigvals = eigvals[sorted_indices]
    sorted_eigvecs = eigvecs[:, sorted_indices]
    sorted_eigvecs[:, 1] /= norm(sorted_eigvecs[:, 1])
    phi= sorted_eigvecs[:, 1] * sorted_eigvecs[:, 1]'
    return phi
end

function vecteur_propre(A) #Matrice des vecteurs propres triés
     eigvals, eigvecs = eigen(A)
    sorted_indices = sortperm(eigvals)
    sorted_eigvals = eigvals[sorted_indices]
    sorted_eigvecs = eigvecs[:, sorted_indices]
    n = size(A, 1)   
    for i in 1:n
        sorted_eigvecs[:, i] /= norm(sorted_eigvecs[:, i])
    end
    sorted_eigvecs
end

function base_propre(A)  #Matrice de la base des vecteurs propres de l'éspace tangent
    sorted_eigvecs = vecteur_propre(A)
    for i in 2:n
        phi[:, i-1] = reshape(sorted_eigvecs[:, i] * (sorted_eigvecs[:, 1])' + sorted_eigvecs[:, 1] * (sorted_eigvecs[:, i])', n^2)
    end
    return phi
end

function retraction(P) #fonction de la retraction
    eigvals, eigvecs = eigen(P)
    D = diagm(eigvals)
    phi=eigvecs
    for j in 1:size(P, 1)
        if abs(D[j, j]) > 0.5
            D[j, j] = 1.0
        else
            D[j, j] = 0.0
        end
    end

    P_retracted = phi * D * phi'
    return P_retracted
end

end # module
