module Maplineair
include("fonction.jl")

function omega(X::AbstractMatrix{T}, P::AbstractMatrix{T}, H::AbstractMatrix{T}) where T #Operateur matricielle omega
    X = projection(P, X)
    om = P * X * (I - P) * H - H * P * X * (I - P)
    omega = om + om'
    return projection(P, omega)
end
#############################################################################################
function K(P::AbstractMatrix{T}, X::AbstractMatrix{T},alpha,dx) where T #Opérateur de la matrice hessienn
    X=projection(P,X)
    k=(alpha/dx)*diagm(diag(X))
    k=projection(P,k)
    return k
end
#############################################################################################
function omega_maps(P_star,H)    #lineaire map omega
    n = size(P_star, 1)
    omega_func = x -> begin
        X = reshape(x, n, n)
        X = projection(P_star, X)
        omega_k = omega(X, P_star, H)
        omega_k = projection(P_star,omega_k)
        vec(omega_k)  
    end
    
    omeg = LinearMap(omega_func, n^2)
    return omeg
end
##############################################################################################
function J_map(P_star,H)    #fonction qui calcul la lineair map de la jacobienne du gradient descent et SCF
    n = size(P_star, 1)
    J_gradient = x -> begin
        X = reshape(x, n, n)
        X = projection(P_star, X)
        J_K = omega(X, P_star, H) + K(P_star, X,alpha,dx)
        J_K = projection(P_star,J_K)
        vec(J_K)  # Convertir la matrice résultante en vecteur
    end  
    omega_map = omega_maps(P_star,H) 
    J_gradient_map = LinearMap(J_gradient, n^2)
    J_SCF = x -> begin
        X = reshape(x, n, n)
        K_X = K(P_star, X, alpha, dx)
        K_X_vec = vec(K_X)
        y=gmres(omega_map, K_X_vec)
        Y=reshape(y,n,n)
        y=vec(projection(P_star,Y))
        y = y + x  
    end
    J_map_SCF = LinearMap(J_SCF, n^2)
    return J_map_SCF,J_gradient_map
end

##############################################################################################
function J_base_réduite(P_star,H) #la lineairmap de la jacobien de SCF réduite dans la base de la tangente
    base=base_propre(H)
    J_map_SCF=J_map(P_star,H)[1]
    ph=vecteur_propre(H)
    J_base_func = x -> begin
        x = reshape(x, n-1)
        X = sum(x[i]*reshape(base[:,i],n,n) for i in 1:n-1)
        x=vec(X)
        x=J_map_SCF*x
        X=reshape(x,n,n)
        @assert norm(projection(P_star, X)-X)<1e-7
        y = [(ph[:,1])'*X*ph[:,i] for i in 2:n]
    end
    J_base=LinearMap(J_base_func,n-1)
    return J_base
end
###############################################################################################
function J_base_réduite_gradient(P_star,H)  #la lineairmap de la jacobien de gradient réduite dans la base de la tangente
    base=base_propre(H)
    J_map_SCF=J_map(P_star,H)[2]
    ph=vecteur_propre(H)
    J_base_func = x -> begin
        x = reshape(x, n-1)
        X = sum(x[i]*reshape(base[:,i],n,n) for i in 1:n-1)
        x=vec(X)
        x=J_map_SCF*x
        X=reshape(x,n,n)
        @assert norm(projection(P_star, X)-X)<1e-7
        y = [(ph[:,1])'*X*ph[:,i] for i in 2:n]
    end
    J_base=LinearMap(J_base_func,n-1)
    return J_base
end
##############################################################################################
function beta_optim(A,n)   # beta optimal sans faire la projection sur la base (boucle sur la premiere valeur propre non nul)
    lambda_max_list = eigs(A, nev=1, which=:LR, tol=1e-10,maxiter=100000)[1]
    lambda_max=norm(lambda_max_list)
    lambda_min_list = eigs(A, nev=n, which=:SR, tol=1e-10,maxiter=10000)[1]
    lambda_min=0
    for i in 1:length(lambda_min_list)
        if norm(lambda_min_list[i])>1+1e-3
            lambda_min=norm(lambda_min_list[i])
            break
        end
    end
    beta_optimal=2/(lambda_max+lambda_min)
end
##############################################################################################
function beta_op(A)   # beta optimal sans faire la projection sur la base (boucle sur la premiere valeur propre non nul)
    lambda_max_list = eigs(A, nev=1, which=:LR, tol=1e-10,maxiter=100000)[1]
    lambda_max=norm(lambda_max_list)
    lambda_min_list = eigs(A, nev=1, which=:SR, tol=1e-10,maxiter=10000)[1]
    lambda_min=norm(lambda_max_list)
    beta_optimal=2/(lambda_max+lambda_min)
end

end #module
