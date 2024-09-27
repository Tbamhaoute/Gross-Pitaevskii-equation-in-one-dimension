using LinearAlgebra
using LinearMaps
using Plots
using SparseArrays
using MatrixMarket
using Statistics
using KernelDensity
using StatsBase
using Base.Threads
using GLM
using DataFrames

function gradient_random2(A, b,Nb,max_iter, tol,σ,case)
    x_old = zeros(Nb)
    x_new = zeros(Nb)
    i = 1
    iter = []
    erreur_list = []
    beta_list=[]
    α_list=[]
    while i < max_iter
        rk = b - A * x_old
        βk = tr(rk'*A*rk) / tr((A*rk)' * (A * rk))
        if case==0
           θk=1 
        elseif case==1
        θk= (1+σ-2*σ*rand())
        #βk= θk*βk\
        elseif case==2
        θk=rand(Normal(1,σ/sqrt(3)))
        #βk= θk*βk
        end
        α=((tr(rk'*A*rk))^2 / ((norm(rk)^2)*norm(A * rk)^2))
        x_new = x_old + θk*βk * rk 

        push!(α_list,α)
        push!(beta_list,βk)
        push!(iter, i)
        r_new = b - A * x_new
        erreur = norm(r_new)
        push!(erreur_list, erreur)

        if erreur < tol
            break
        end

        x_old = x_new
        i += 1
    end

    return x_new, iter, erreur_list,beta_list,α_list
end

function construction(Nb) #construction du hamiltonien
    x = LinRange(0, 1, Nb)
    dx = 1 / Nb
    main_diag = fill((1 / dx^2),Nb)
    off_diag = fill(-1 / (2 * dx^2), Nb - 1)

    h = spdiagm(0 => main_diag, 1 => off_diag, -1 => off_diag)
    return Matrix(h)
end

# Construction des matrices A et b
Nb=10
A=construction(Nb)                               #matrice laplacien
#A = MatrixMarket.mmread("bcsstk01.mtx")         #matrice généré avec Matrix market
A = Matrix(A)
b=ones(Nb)
eigenvalues=real.(eigvals(A));

##############################################################################################################
# Résultats pour loi uniforme 
max_iter = 10000
tol = 1e-6
σ_list = collect(0.1:0.1:1)
n = 1000
sol, iter_list, erreur_list, β_list,α_list = gradient_random2(A, b, Nb, max_iter, tol, 0,1)
T = Array{Any}(undef, n, length(σ_list), 4)
for i in 1:length(σ_list)
    @threads for j in 1:n
        sol2, iter_list2, erreur_list2, β_list2,α_list2 = gradient_random2(A, b, Nb, max_iter, tol, σ_list[i],1)
        T[j, i, 1] = iter_list2[end] /iter_list[end]
        T[j, i, 2] = iter_list2
        T[j, i, 3] = erreur_list2
        
        log_y = log10.(erreur_list2)
        df = DataFrame(x=iter_list2, log_y=log_y)
        model = lm(@formula(log_y ~ x), df)
        bias = coef(model)[1]
        pente = coef(model)[2]
        x_n = bias .+ pente .* iter_list2
        y_n = 10 .^(x_n)
        T[j, i, 4] = pente,bias,y_n
    end
end

histogram(T[:,2,1],label="σ=$(σ_list[2])",title="Loi uniforme")
for i in 3:length(σ_list)
    histogram!(T[:,i,1],label="σ=$(σ_list[i])")
end
p1=histogram!()

list_écart=[]
list_var=[]
for i in 1:length(σ_list)
    écart_moy =((sum(T[:,i,1])))/n  # Calcul de l'écart moyen
    variance=var(T[:,i,1])
    push!(list_écart,écart_moy)
    push!(list_var,variance)
end
indice=argmin(list_écart)
println("le σ optimal est : $(σ_list[indice])")
plot(σ_list,list_écart,label="écart moyenne",xlabel="σ")
plot!(σ_list,list_var,label="variance",xlabel="σ")
p2=scatter!([(σ_list[indice])],[list_écart[indice]],label="σ optimal = $(σ_list[indice])")

display([p1,p2])
