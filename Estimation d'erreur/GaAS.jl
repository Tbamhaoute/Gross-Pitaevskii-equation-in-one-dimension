using Pkg
Pkg.instantiate()

using DFTK
using Printf
using LinearAlgebra
using ForwardDiff
using LinearMaps
using IterativeSolvers
using Plots
using Infiltrator
using JLD2

a = 10.68290949909  # GaAs lattice constant in Bohr
lattice = a / 2 * [[0 1 1.];
                   [1 0 1.];
                   [1 1 0.]]

Ga = ElementPsp(:Ga, psp=load_psp("hgh/lda/ga-q3"))
As = ElementPsp(:As, psp=load_psp("hgh/lda/as-q5"))
atoms = [Ga, As]
positions = [ones(3)/8 + [0.24, -0.33, 0.12] / 15, -ones(3)/8]

model = model_LDA(lattice, atoms, positions)
kgrid = [3, 3, 3]  # k-point grid (Regular Monkhorst-Pack grid)
ss = 4
Ecut_ref=500
fft_size_ref = compute_fft_size(model, Ecut_ref; supersampling=ss)
basis_ref = PlaneWaveBasis(model; Ecut=Ecut_ref, kgrid, fft_size=fft_size_ref)
tol = 1e-13

scfres_ref = self_consistent_field(basis_ref; tol,
                                   is_converged=DFTK.ScfConvergenceDensity(tol))
f_ref = compute_forces(scfres_ref)
ψ_ref, occupation = DFTK.select_occupied_orbitals(basis_ref, scfres_ref.ψ, scfres_ref.occupation)
E_ref=scfres_ref.energies.total


    #fonction pour appliquer la matrice M
    function apply_M(φk, Pk, δφnk, n)
        DFTK.proj_tangent_kpt!(δφnk, φk)
        δφnk = sqrt.(Pk.mean_kin[n] .+ Pk.kin) .* δφnk
        DFTK.proj_tangent_kpt!(δφnk, φk)
        δφnk = sqrt.(Pk.mean_kin[n] .+ Pk.kin) .* δφnk
        DFTK.proj_tangent_kpt!(δφnk, φk)
    end
    #fonction pour appliquer L'inverse de la matrice M
    function apply_inv_M(φk, Pk, δφnk, n)
        DFTK.proj_tangent_kpt!(δφnk, φk)
        op(x) = apply_M(φk, Pk, x, n)
        function f_ldiv!(x, y)
            x .= DFTK.proj_tangent_kpt(y, φk)
            x ./= (Pk.mean_kin[n] .+ Pk.kin)
            DFTK.proj_tangent_kpt!(x, φk)
        end
        J = LinearMap{eltype(φk)}(op, size(δφnk, 1))
        δφnk = cg(J, δφnk; Pl=DFTK.FunctionPreconditioner(f_ldiv!),
                  verbose=false, reltol=0, abstol=1e-15)
        DFTK.proj_tangent_kpt!(δφnk, φk)
    end
    
    function apply_metric(φ, P, δφ, A::Function)
        map(enumerate(δφ)) do (ik, δφk)
            Aδφk = similar(δφk)
            φk = φ[ik]
            for n = 1:size(δφk,2)
                Aδφk[:,n] = A(φk, P[ik], δφk[:,n], n)
            end
            Aδφk
        end
    end
    #fonction pour calculer le differentiel en utilisant la fonction ForwardDiff
    function df(basis, occupation, ψ, δψ, ρ)
        δρ = DFTK.compute_δρ(basis, ψ, δψ, occupation)
        ForwardDiff.derivative(ε -> compute_forces(basis, ψ.+ε.*δψ, occupation; ρ=ρ+ε.*δρ), 0)
    end;
        #fonction pour appliquer la matrice M
        function apply_M(φk, Pk, δφnk, n)
            DFTK.proj_tangent_kpt!(δφnk, φk)
            δφnk = sqrt.(Pk.mean_kin[n] .+ Pk.kin) .* δφnk
            DFTK.proj_tangent_kpt!(δφnk, φk)
            δφnk = sqrt.(Pk.mean_kin[n] .+ Pk.kin) .* δφnk
            DFTK.proj_tangent_kpt!(δφnk, φk)
        end
        #fonction pour appliquer L'inverse de la matrice M
        #fonction pour calculer l'erreur en alligniant les deux solutions
    function compute_error(basis, ϕ, ψ)
        map(zip(ϕ, ψ)) do (ϕk, ψk)
            S = ψk'ϕk
            U = S*(S'S)^(-1/2)
            ϕk - ψk*U
        end
    end

    function gamma_norm(basis,scfres,Ecut_init,x,case)
        basis_gamma = PlaneWaveBasis(model; Ecut=Ecut_init*x, kgrid)
       #projection de  ψ_ref dans la base gamma
       ψ_ref_gamma = DFTK.transfer_blochwave(ψ_ref,basis_ref,basis_gamma);
       ψ_ref_gamma = DFTK.select_occupied_orbitals(basis_gamma, ψ_ref_gamma, scfres_ref.occupation).ψ;
       ψ_gamma = DFTK.transfer_blochwave(scfres.ψ, basis, basis_gamma);
       ρr = compute_density(basis_gamma, ψ_gamma, scfres.occupation);
       Er, hamr = energy_hamiltonian(basis_gamma, ψ_gamma, scfres.occupation; ρ=ρr);  
       res = DFTK.compute_projected_gradient(basis_gamma, ψ_gamma, scfres.occupation);
       res, occ = DFTK.select_occupied_orbitals(basis_gamma, res, scfres.occupation);
       ψ_gamma = DFTK.select_occupied_orbitals(basis_gamma, ψ_gamma, scfres.occupation).ψ;
       ham_basis=energy_hamiltonian(basis, scfres.ψ, scfres.occupation; ρ=scfres.ρ)[2];
       P = [PreconditionerTPA(basis_gamma, kpt) for kpt in basis_gamma.kpoints]
       map(zip(P, ψ_gamma)) do (Pk, ψk)
           DFTK.precondprep!(Pk, ψk)
       end
        # Rayleigh coefficients needed for apply_Ω
       Λ = map(enumerate(ψ_gamma)) do (ik, ψk)
           Hk = hamr.blocks[ik]
           Hψk = Hk * ψk
           ψk'Hψk
       end
       
       if case==1
           Mres = apply_metric(ψ_gamma, P, res, apply_inv_M);
           resLF = DFTK.transfer_blochwave(res, basis_gamma, basis);
           resHF = res - DFTK.transfer_blochwave(resLF, basis, basis_gamma);
           e2 = apply_metric(ψ_gamma, P, resHF, apply_inv_M);   
           M22_e2=apply_metric(ψ_gamma, P, e2, apply_M);   
           J_e2=DFTK.apply_Ω(e2, ψ_gamma, hamr, Λ) .+ DFTK.apply_K(basis_gamma, e2, ψ_gamma, ρr, occ);
           J_e2_1 = DFTK.transfer_blochwave(J_e2, basis_gamma, basis);
           J_e2_1 = DFTK.transfer_blochwave(J_e2_1, basis, basis_gamma);
           #calcul de la prohection dans la base orthogonale (haute fréquences) proj(Je2)
           J22_e2 = J_e2-J_e2_1
           return norm(M22_e2-J22_e2)
       end 
       if case==2
           e = compute_error(basis_gamma, ψ_gamma, ψ_ref_gamma);
           e1=DFTK.transfer_blochwave(e, basis_gamma, basis);
           e1=DFTK.transfer_blochwave(e1, basis, basis_gamma);
           e2=e-e1
           M22_e2=apply_metric(ψ_gamma, P, e2, apply_M);   
           #calcul de (Ω+K)*x1
           J_e1=DFTK.apply_Ω(e1, ψ_gamma, hamr, Λ) .+ DFTK.apply_K(basis_gamma, e1, ψ_gamma, ρr, occ);
           #calcul de la projection dans la petite base
           J_e1_1 = DFTK.transfer_blochwave(J_e1, basis_gamma, basis);
           J_e1_1 = DFTK.transfer_blochwave(J_e1_1, basis, basis_gamma);
           #calcul de la prohection dans la base orthogonale (haute fréquences)
           J21_e1 = J_e1-J_e1_1
       
           J_e2=DFTK.apply_Ω(e2, ψ_gamma, hamr, Λ) .+ DFTK.apply_K(basis_gamma, e2, ψ_gamma, ρr, occ);
           J_e2_1 = DFTK.transfer_blochwave(J_e2, basis_gamma, basis);
           J_e2_1 = DFTK.transfer_blochwave(J_e2_1, basis, basis_gamma);
           #calcul de la prohection dans la base orthogonale (haute fréquences) proj(Je2)
           J22_e2 = J_e2-J_e2_1
           s=J22_e2.+J21_e1
           #N_gamma=norm(resHF)/norm(s)
           #N_gamma=norm(resHF.-J22_e2)
           return norm(M22_e2-s)
       end 
   end
   

   function erreur_energy2(basis,scfres,Ecut_init,x)
    basis_gamma = PlaneWaveBasis(model; Ecut=Ecut_init*x, kgrid)
   ψr = DFTK.transfer_blochwave(scfres.ψ, basis, basis_gamma)
   ρr = compute_density(basis_gamma, ψr, scfres.occupation)
   Er, hamr = energy_hamiltonian(basis_gamma, ψr, scfres.occupation; ρ=ρr);
   res = DFTK.compute_projected_gradient(basis_gamma, ψr, scfres.occupation)
   res, occ = DFTK.select_occupied_orbitals(basis_gamma, res, scfres.occupation)
   ψr = DFTK.select_occupied_orbitals(basis_gamma, ψr, scfres.occupation).ψ;
   P = [PreconditionerTPA(basis_gamma, kpt) for kpt in basis_gamma.kpoints]
   map(zip(P, ψr)) do (Pk, ψk)
       DFTK.precondprep!(Pk, ψk)
   end
   
   
   Mres = apply_metric(ψr, P, res, apply_inv_M);
   
   resLF = DFTK.transfer_blochwave(res, basis_gamma, basis)
   resHF = res - DFTK.transfer_blochwave(resLF, basis, basis_gamma);
   e2 = apply_metric(ψr, P, resHF, apply_inv_M);
   # Rayleigh coefficients needed for `apply_Ω`
   Λ = map(enumerate(ψr)) do (ik, ψk)
       Hk = hamr.blocks[ik]
       Hψk = Hk * ψk
       ψk'Hψk
   end
   ΩpKe2 = DFTK.apply_Ω(e2, ψr, hamr, Λ) .+ DFTK.apply_K(basis_gamma, e2, ψr, ρr, occ)
   ΩpKe2 = DFTK.transfer_blochwave(ΩpKe2, basis_gamma, basis)
   rhs = resLF - ΩpKe2;
   (; ψ) = DFTK.select_occupied_orbitals(basis, scfres.ψ, scfres.occupation)
   e1 = DFTK.solve_ΩplusK(basis, ψ, rhs, occ; tol).δψ
   e1 = DFTK.transfer_blochwave(e1, basis, basis_gamma)
   
   res_schur = e1 + Mres
   ψ_erreur=ψr-res_schur
   ρr = compute_density(basis_gamma, ψ_erreur, occ)
   E_approché=energy_hamiltonian(basis_gamma, ψ_erreur, occ; ρ=ρr)[1].total
   E_approché,norm(E_approché- E_ref)
end



function detecter_stagnation(f, a, b; epsilon=1e-1, tol_ratio=1e-2)
    while (b - a) > epsilon
        c = (a + b) / 2
        delta_f = abs(f(c + epsilon) - f(c-epsilon))/abs(2*f(c))

        if delta_f < tol_ratio
            b = c
        else
            a = c
        end
    end
    return (a + b) / 2
end

##############################################################################################################
Ecut_init = [35, 50,60,75,100,120,150]
Ecut_list = collect(1:0.2:3)

gamma_list = []
f_liste = []
g_liste = []

data_courbes_erreur = Dict()
data_courbes_gamma = Dict()

# Boucle combinée sur les valeurs d'Ecut et de gamma
for i in 1:length(Ecut_init)
    Ecut = Ecut_init[i]
    base_petite = PlaneWaveBasis(model; Ecut=Ecut, kgrid=kgrid)
    scfres = self_consistent_field(base_petite; tol=tol, callback=identity)
    
    # Définition des fonctions f et g
    f(x) = erreur_energy2(base_petite, scfres, Ecut, x)[2]
    g(x) = gamma_norm(base_petite, scfres, Ecut, x, 1)
    
    # Détection des points de stagnation
    gamma2 = detecter_stagnation(g, 1.1, 4)
    
    push!(gamma_list, gamma2)
    push!(f_liste, f(gamma2))
    push!(g_liste, g(gamma2))
    
    err = []
    gamm = []
    
    # Calcul des erreurs pour chaque valeur de gamma dans Ecut_list
    for j in 1:length(Ecut_list)
        er = erreur_energy2(base_petite, scfres, Ecut, Ecut_list[j])[2]
        gamma_val = gamma_norm(base_petite, scfres, Ecut, Ecut_list[j], 1)
        
        push!(err, er)
        push!(gamm, gamma_val)
    end
    
    # Stockage des résultats
    data_courbes_erreur["Ecut=$(Ecut)"] = (
        Ecut_list = Ecut_list, 
        err = err, 
    )
    
    data_courbes_gamma["Ecut=$(Ecut)"] = (
        Ecut_list = Ecut_list, 
        gamm = gamm, 
    )
end

# Sauvegarde des données
@save "data_erreur_GaAS.jld2" data_courbes_erreur data_courbes_gamma gamma_list f_liste g_liste

# graphique
plot(xlabel="gamma", ylabel="Erreur d'énergie", title="Variation de l'erreur d'énergie en fonction de gamma")

# Tracé des courbes d'erreur pour chaque valeur de Ecut
for (key, value) in data_courbes_erreur
    Ecut_list = value[:Ecut_list]
    err = value[:err]
    plot!(Ecut_list, err, yscale=:log10, label=key)  # Ajouter les labels pour chaque courbe
end

# Ajout des points de stagnation
plot!(gamma_list, f_liste, seriestype=:scatter, marker=:circle, color=:red, label="Points de stagnation")
savefig("energie_GaAs.pdf")
plot(xlabel="Gamma", ylabel="Erreur du norme", title="Variation de l'erreur du norme en fonction de Gamma", legend=:topright)

for (key, value) in data_courbes_gamma
    Ecut_list = value[:Ecut_list]
    gamm = value[:gamm]
    plot!(Ecut_list, gamm, label=key)  
end
plot!(gamma_list, g_liste, seriestype=:scatter, marker=:circle, color=:red, label="Points de stagnation")
savefig("norme_GaAs.pdf")
