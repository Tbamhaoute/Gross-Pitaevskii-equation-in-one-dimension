include("fonction.jl")
include("Point_fixe.jl")
include("Gradient.jl")
include("Maplineair.jl")
using .fonction 
using .Point_fixe
using .Gradient
using .Maplineair
using LinearAlgebra
using Printf
using Plots




alpha = 50
max_iter = 4000
Nb=100
dx=1/Nb
tolerance = 1e-10
##################################################################################
h=construction(Nb)
P_init=phi(h)
P_star=SCF(Nb, alpha, 0.1, 5000, 1e-12)[1]
H = h + (alpha / dx) * diagm(diag(P_star))
n = size(P_star, 1)
J_map_SCF=J_map(P_star,H,alpha,dx)[1]
J_map_grad=J_map(P_star,H,alpha,dx)[2]
J_base_réd_grad=J_base_réduite_gradient(P_star,H,alpha,dx) #jacobienne du gradient descent projeté dans l'espace tangent
J_base_réd=J_base_réduite(P_star,H,alpha,dx)  #jacobienne de la SCF projeté dans l'espace tangent
J_mat_scf=Matrix(J_base_réd) #convertir en matrice

lambda_max,lambda_min,beta_optimal=beta_op(J_base_réd_grad)
lambda_max_scf,lambda_min_scf,beta_optimal_scf=beta_op(J_mat_scf)


title = "gradient descent"
Pinit=phi(h)

time1 = @elapsed P_final, iterations, erreurs = gradient_descent(Nb, alpha, beta_optimal, max_iter, tolerance)
time2 = @elapsed P_final2, iterations2, erreurs2 = gradient_descent(Nb, alpha, 5e-5, max_iter, tolerance)
time3 = @elapsed P_final3, iterations3, erreurs3 = gradient_barzilai(Nb, alpha, 5e-5, max_iter, tolerance)
time4 = @elapsed P_final4, iterations4, erreurs4 = gradient_descent_variable(Nb, alpha, max_iter, tolerance)
time5 = @elapsed P_final5, iterations5, erreurs5 = gradient_descent_random(Nb, alpha, max_iter, tolerance)


times = [time1, time2, time3, time4, time5]
iterations_list = [iterations[end], iterations2[end], iterations3[end], iterations4[end], iterations5[end]]

# temps de calcul
println("Methode                         | Temps (s)  | Iterations")
println("------------------------------------------------------")
println(@sprintf("gradient_descent (beta optimal)| %8.4f  | %10d", times[1], iterations_list[1]))
println(@sprintf("gradient_descent (beta=5e-5)   | %8.4f  | %10d", times[2], iterations_list[2]))
println(@sprintf("gradient_barzilai              | %8.4f  | %10d", times[3], iterations_list[3]))
println(@sprintf("gradient_descent_variable      | %8.4f  | %10d", times[4], iterations_list[4]))
println(@sprintf("gradient_descent_random        | %8.4f  | %10d", times[5], iterations_list[5]))

# Plot des results
plot(iterations, erreurs, xlabel="Nombre d'itérations", ylabel="Erreur", title="Convergence de l'algorithme $title", label="beta_optimal", yscale=:log10, color=1, lw=2)
plot!(iterations2, erreurs2, xlabel="Nombre d'itérations", ylabel="Erreur", label="beta=5e-5", yscale=:log10, color=2, lw=2)
plot!(iterations3, erreurs3, xlabel="Nombre d'itérations", ylabel="Erreur", label="Barzilai-Borwein", yscale=:log10, lw=2)
plot!(iterations4, erreurs4, xlabel="Nombre d'itérations", ylabel="Erreur", label="gradient descent à pas variable", yscale=:log10, lw=2)
plot!(iterations5, erreurs5, xlabel="Nombre d'itérations", ylabel="Erreur", label="gradient descent random", yscale=:log10, lw=2)

#  asymptotic rates
r_opt = max(abs(1 - beta_optimal * lambda_max), abs(1 - beta_optimal * lambda_min))
r_1 = max(abs(1 - (5e-5) * lambda_max), abs(1 - (5e-5) * lambda_min))
YY_opt = [0.005 * r_opt^(i) for i in 500:1:1000]
YY_1 = [0.005 * r_1^i for i in 500:1:1000]
plot!(500:1:1000, YY_opt, xlabel="itérations", ylabel="assymptotique rate", label="assymptotic rate beta_optimal", yscale=:log10, color=1, linestyle=:dash, lw=2)
plot!(500:1:1000, YY_1, xlabel="itérations", ylabel="Erreur", label="assymptotic rate beta=5e-5", yscale=:log10, color=2, legendfontsize=10, linestyle=:dash, lw=2)
savefig("Gradient.pdf")

Nb = 100
dx = 1 / Nb
alpha = 50
beta=0.1
max_iter = 1000
tolerance = 1e-10
time1 = @elapsed P_final, iterations, erreurs = SCF(Nb, alpha, beta, max_iter, tolerance)
time2 = @elapsed P_final_opt, iterations_opt, erreurs_opt = SCF(Nb, alpha, beta_optimal_scf, max_iter, tolerance)
time3 = @elapsed P_final2, iterations2, erreurs2 = SCF_pas_variable(Nb, alpha, max_iter, tolerance)
time4 = @elapsed P_final3, iterations3, erreurs3 = SCF_pas_variable2(Nb, alpha, max_iter, tolerance)
time5 = @elapsed P_final4, iterations4, erreurs4 = SCF_pas_random(Nb, alpha, max_iter, tolerance)

times = [time1, time2, time3, time4, time5]
iterations_counts = [length(iterations), length(iterations_opt), length(iterations2), length(iterations3), length(iterations4)]

# temps de calcul
println("Method                  | Temps (s)  | Iterations")
println("-----------------------------------------------")
println(@sprintf("SCF (beta=0.1)          | %8.4f  | %10d", times[1], iterations_counts[1]))
println(@sprintf("SCF (beta_optimal)      | %8.4f  | %10d", times[2], iterations_counts[2]))
println(@sprintf("SCF_pas_variable        | %8.4f  | %10d", times[3], iterations_counts[3]))
println(@sprintf("SCF_pas_variable2       | %8.4f  | %10d", times[4], iterations_counts[4]))
println(@sprintf("SCF_pas_random          | %8.4f  | %10d", times[5], iterations_counts[5]))

# Plot des results
plot(iterations, erreurs, xlabel="Nombre d'itérations", ylabel="Erreur", label="beta=0.1", title="Convergence de l'algorithme SCF", yscale=:log10, lw=2, color=2)
plot!(iterations_opt, erreurs_opt, xlabel="Nombre d'itérations", ylabel="Erreur", label="beta=beta_optimal", title="Convergence de l'algorithme SCF", yscale=:log10, lw=2, color=1)
plot!(iterations2, erreurs2, xlabel="Nombre d'itérations", ylabel="Erreur", label="beta variable", yscale=:log10, lw=2)
plot!(iterations3, erreurs3, xlabel="Nombre d'itérations", ylabel="Erreur", label="beta variable2", yscale=:log10, lw=2)
plot!(iterations4, erreurs4, xlabel="Nombre d'itérations", ylabel="Erreur", label="beta random", yscale=:log10, lw=2)

#  asymptotic rates
r_opt = max(abs(1 - beta_optimal_scf * lambda_max_scf), abs(1 - beta_optimal_scf * lambda_min_scf))
r_1 = max(abs(1 - (0.1) * lambda_max_scf), abs(1 - (0.1) * lambda_min_scf))
YY_opt = [0.005 * r_opt^(i) for i in 45:1:60]
YY_1 = [0.0015 * r_1^i for i in 80:1:90]
plot!(45:1:60, YY_opt, xlabel="itérations", ylabel="assymptotique rate", label="assymptotic rate beta_optimal", yscale=:log10, color=1, linestyle=:dash, lw=2)
plot!(80:1:90, YY_1, xlabel="itérations", ylabel="Erreur", label="assymptotic rate beta=0.1", yscale=:log10, color=2, legendfontsize=10, linestyle=:dash, lw=2)
savefig("SCF.pdf")