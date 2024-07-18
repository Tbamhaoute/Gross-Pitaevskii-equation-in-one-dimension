module Point_fixe
using LinearAlgebra
include("fonction.jl") 
using .fonction
export SCF,SCF_pas_random,SCF_pas_variable,SCF_pas_variable2

function SCF(Nb, alpha, beta, max_iter, tolerance)  # point fixe avec beta fixe
h=construction(Nb)
dx=1/Nb
P_old=phi(h)
   #= P_perturb=Symmetric(rand(Nb,Nb))     pour ajouter une perturbation à la solution initial
    P_perturb/=10*norm(P_perturb)
    P_old=retraction(P_old+P_perturb)=#
s = []
erreur = []

for k in 1:max_iter
    H = h + (alpha / dx) * diagm(diag(P_old))
    P_new = P_old + beta * projection(P_old, phi(H) - P_old)
    P_new=Symmetric(P_new)
    P_new=retraction(P_new)
    push!(s,k)
    push!(erreur, norm(P_new-P_old))
    # Vérification de la convergence
    if norm(P_new-P_old) < tolerance
        #println("Convergence atteinte après $k itérations.")
            return P_old, s, erreur
                    break

    end
    P_old=P_new
end
#println("Nombre maximum d'itérations atteint.")
    return P_old, s, erreur
end

function SCF_pas_variable(Nb, alpha, max_iter, tolerance)   #point fixe à pas variable avec méthode de la trace
h=construction(Nb)
dx=1/Nb
P_old=phi(h)
   #= P_perturb=Symmetric(rand(Nb,Nb))
    P_perturb/=10*norm(P_perturb)
    P_old=retraction(P_old+P_perturb) =#
s = []
erreur = []
for k in 1:max_iter
    H = h + (alpha / dx) * diagm(diag(P_old))
    F_k=projection(P_old, phi(H) - P_old)
    #=    H_k=h + (alpha / dx) * diagm(diag(F_k))
    F_2k=phi(F_k)
    delta_k=F_2k-2*F_k+P_old
    beta=tr(delta_k*(F_k-P_old)')/norm(delta_k)^2=#
        beta=-(tr(h*F_k*P_old)+(alpha/(2*dx))*sum(P_old[i,i]*F_k[i,i] for i in 1:Nb))/(tr(h*F_k^2)+(alpha/(2*dx))*sum(F_k[i,i]^2 for i in 1:Nb))
    P_new = P_old + beta * projection(P_old, phi(H) - P_old)
    P_new=Symmetric(P_new)
    P_new=retraction(P_new)
    push!(s,k)
    push!(erreur, norm(P_new-P_old))
    # Vérification de la convergence
    if norm(P_new-P_old) < tolerance
        #println("Convergence atteinte après $k itérations.")
            return P_old, s, erreur
                    break

    end
    P_old=P_new
end
#println("Nombre maximum d'itérations atteint.")
    return P_old, s, erreur
end

function SCF_pas_variable2(Nb, alpha, max_iter, tolerance)  #point fixe à pas variable avec méthode du double diagonalisation
h=construction(Nb)
dx=1/Nb
P_old=phi(h)
   #= P_perturb=Symmetric(rand(Nb,Nb))
    P_perturb/=10*norm(P_perturb)
    P_old=retraction(P_old+P_perturb) =#
s = []
erreur = []
for k in 1:max_iter
    H = h + (alpha / dx) * diagm(diag(P_old))
    F_k=phi(H)
        H_k=h + (alpha / dx) * diagm(diag(F_k))
    F_2k=phi(H_k)
    delta_k=F_2k-2*F_k+P_old
    beta=-tr(delta_k*(F_k-P_old)')/norm(delta_k)^2
    P_new = P_old + beta * projection(P_old, phi(H) - P_old)
    P_new=Symmetric(P_new)
    P_new=retraction(P_new)
    push!(s,k)
        H_new=h + (alpha / dx) * diagm(diag(P_new))
    push!(erreur, norm(P_new-phi(H_new)))
    # Vérification de la convergence
    if norm(P_new-P_old) < tolerance
        #println("Convergence atteinte après $k itérations.")
            return P_old, s, erreur
                    break

    end
    P_old=P_new
end
#println("Nombre maximum d'itérations atteint.")
    return P_old, s, erreur
end

function SCF_pas_random(Nb, alpha, max_iter, tolerance)  # multiplication du beta variable 1 par une loi uniforme entre [0.5,1.5]
h=construction(Nb)
dx=1/Nb
P_old=phi(h)
   #= P_perturb=Symmetric(rand(Nb,Nb))
    P_perturb/=10*norm(P_perturb)
    P_old=retraction(P_old+P_perturb) =#
s = []
erreur = []
for k in 1:max_iter
    H = h + (alpha / dx) * diagm(diag(P_old))
    F_k=projection(P_old, phi(H) - P_old)
        beta=-(0.5+rand())*(tr(h*F_k*P_old)+(alpha/(2*dx))*sum(P_old[i,i]*F_k[i,i] for i in 1:Nb))/(tr(h*F_k^2)+(alpha/(2*dx))*sum(F_k[i,i]^2 for i in 1:Nb))
    P_new = P_old + beta * projection(P_old, phi(H) - P_old)
    P_new=Symmetric(P_new)
    P_new=retraction(P_new)
    push!(s,k)
    push!(erreur, norm(P_new-P_old))
    # Vérification de la convergence
    if norm(P_new-P_old) < tolerance
        #println("Convergence atteinte après $k itérations.")
            return P_old, s, erreur
                    break

    end
    P_old=P_new
end
#println("Nombre maximum d'itérations atteint.")
    return P_old, s, erreur
end

end #module
