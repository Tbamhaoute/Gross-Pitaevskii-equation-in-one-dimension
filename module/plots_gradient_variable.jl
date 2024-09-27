include("fonction.jl")
include("Point_fixe.jl")
include("Gradient.jl")
include("Maplineair.jl")
include("Gradient2.jl")
using .fonction 
using .Point_fixe
using .Gradient
using .Maplineair
using .Gradient2
using LinearAlgebra
using Printf
using Plots



Nb=100
dx=1/Nb
alpha_val=50
max_iter=5000
tol=1e-10

h=construction(Nb)
P_init=phi(h)
P_star=SCF(Nb, alpha_val, 0.1, 5000, 1e-12)[1]
H = h + (alpha_val / dx) * diagm(diag(P_star))
n = size(P_star, 1)
J_map_grad=J_map(P_star,H,alpha_val,dx)[2]


P_j,s_j,erreur_j,beta_list_j=gradient_j(Nb, alpha_val, max_iter, 1e-10,J_map_grad,1);
P_j2,s_j2,erreur_j2,beta_list_j2=gradient_j(Nb, alpha_val, max_iter, 1e-10,J_map_grad,2);
P_0,s0,er0,b0=gradient_variable(Nb, alpha_val, max_iter, 1e-10, 3) 
P_1,s1,er1,b1=gradient_variable(Nb, alpha_val, max_iter, 1e-10, 4) 
P_2,s2,er2,b2=gradient_variable(Nb, alpha_val, max_iter, 1e-10, 5) 
P_3,s3,er3,b3=gradient_variable(Nb, alpha_val, max_iter, 1e-10, 1) 
P_4,s4,er4,b4=gradient_variable(Nb, alpha_val, max_iter, 1e-10, 2) 

plot(s0,er0,yscale=:log10,label="β=5e-5")
plot!(s1,er1,yscale=:log10,label="β variable 1")
plot!(s2,er2,yscale=:log10,label="β variable 1 avec loi uniforme",legendfontsize=11)
plot!(s_j,erreur_j,yscale=:log10,label="β variable 2",xlabel="iterations")
plot!(s_j2,erreur_j2,yscale=:log10,label="β variable 2 avec loi uniforme",ylabel="Erreur")
plot!(s3,er3,yscale=:log10,ylabel="résidu",xlabel="itération",label="β variable 3")
plot!(s4,er4,yscale=:log10,label="β variable 3 avec loi uniforme")