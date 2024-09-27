module Gradient2
include("fonction.jl")
include("Maplineair.jl") 
using .fonction
using .Maplineair
using LinearAlgebra
using SparseArrays
using Arpack
using Polynomials
using Roots


export gradient_j,gradient_variable

function gradient_j(Nb, alpha_val, max_iterations, tolerance,J_map_grad,case)    
    dx = 1 / Nb
    h = construction(Nb)
    P_old = phi(h)
    P_new = zeros(Nb,Nb)
    s = []
    erreur = []
    beta_list=[]
    
    for k in 1:max_iterations
        H = h + (alpha_val / dx) * diagm(diag(P_old))
        rk=projection(P_old,H)
        Lk=(alpha_val/dx)*diagm(diag(rk))
        J_map_rk=reshape(J_map_grad*vec(rk),Nb,Nb)
        if case ==1
        β=tr(rk'*J_map_rk)/tr(J_map_rk'*J_map_rk)
        else
        β=2*rand()*tr(rk'*J_map_rk)/tr(J_map_rk'*J_map_rk)
        end
        push!(beta_list,β)
        P_new = P_old - β * rk
        P_new=Symmetric(P_new)
        P_new = retraction(P_new)
        push!(s,k)
        push!(erreur,  norm(P_new-P_old))
        # Vérification de la convergence
        if norm(P_new-P_old) < tolerance
            #println("Convergence atteinte après $k itérations.")
                 return P_old, s, erreur,beta_list
            break
        end
        P_old=P_new
    end
return P_old, s, erreur,beta_list
end

function gradient_variable(Nb, alpha_val, max_iterations, tolerance,case) 
    B(r,X,P)=-r*X-X*r+2*P*X*r+2*r*X*P
    C(r,X)=-2*r*X*r
    dx = 1 / Nb
    h = construction(Nb)
    P_old = phi(h)
    P_new = zeros(Nb,Nb)
    s = []
    erreur = []
    beta_list=[]
    for k in 1:max_iterations
        H = h + (alpha_val / dx) * diagm(diag(P_old))
        rk=projection(P_old,H)
        Lk=(alpha_val/dx)*diagm(diag(rk))

        a=-tr(rk'*C(rk,Lk))
        b=tr(rk'*(-B(rk,Lk,P_old) + C(rk,H)))
        c=tr(rk'*(-projection(P_old,Lk) + B(rk,H,P_old)))
        d=tr(rk'*projection(P_old,H))
        p = Polynomial([d, c, b, a]) 
        racines = roots(p)
        racines_reelles = filter(isreal, racines)
        racines_reelles = real(racines_reelles)
        #println(racines_reelles)
        if case==1
            beta = minimum(abs.(racines_reelles)) 
        elseif case==2
            beta = 2*rand()*minimum(abs.(racines_reelles))
        elseif case ==3
            beta=5e-5
        elseif case ==4
            beta=(tr(h*P_old*rk)+(alpha_val/(2*dx))*sum(P_old[i,i]*rk[i,i] for i in 1:Nb))/(tr(h*rk^2)+(alpha_val/(2*dx))*sum(rk[i,i]^2 for i in 1:Nb))
        elseif case ==5
            beta=2*rand()*(tr(h*P_old*rk)+(alpha_val/(2*dx))*sum(P_old[i,i]*rk[i,i] for i in 1:Nb))/(tr(h*rk^2)+(alpha_val/(2*dx))*sum(rk[i,i]^2 for i in 1:Nb))
        end
        push!(beta_list,beta)
        P_new = P_old - beta * rk
        P_new=Symmetric(P_new)
        P_new = retraction(P_new)
        push!(s,k)
        push!(erreur, norm(P_new-P_old))
        # Vérification de la convergence
        if norm(P_new-P_old) < tolerance
            #println("Convergence atteinte après $k itérations.")
                 return P_old, s, erreur,beta_list
            break
        end
        P_old=P_new
    end
return P_old, s, erreur,beta_list
end



end #module 