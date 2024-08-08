using DFTK
using Printf
using LinearAlgebra
using ForwardDiff
using LinearMaps
using IterativeSolvers
using Plots


a = 10.26  
lattice = a / 2 * [[0 1 1.]; [1 0 1.]; [1 1 0.]]
Si = ElementPsp(:Si; psp=load_psp("hgh/lda/Si-q4"))
atoms = [Si, Si]
positions = [ones(3) / 8, -ones(3) / 8 ]
positions[1] .+= [-0.022, 0.028, 0.035]
model = model_LDA(lattice, atoms, positions)
kgrid=[1, 1, 1]
tol=1e-5
#Calcul de la solution de référence 
basis_ref = PlaneWaveBasis(model; Ecut=500, kgrid=[1, 1, 1])
scfres_ref = self_consistent_field(basis_ref; tol=1e-10);
ψ_ref = DFTK.select_occupied_orbitals(basis_ref, scfres_ref.ψ, scfres_ref.occupation).ψ;
f_ref = compute_forces(scfres_ref)
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
#fonction pour calculer l'erreur en alligniant les deux solutions
function compute_error(basis, ϕ, ψ)
    map(zip(ϕ, ψ)) do (ϕk, ψk)
        S = ψk'ϕk
        U = S*(S'S)^(-1/2)
        ϕk - ψk*U
    end
end

#fonction pour calculer l'erreur de l'énérgie en utilisant ForwardDiff
function dE(basis, occupation, ψ, δψ, ρ)
    δρ = DFTK.compute_δρ(basis, ψ, δψ, occupation)
    ForwardDiff.derivative(ε ->energy_hamiltonian(basis, ψ.+ε.*δψ, occupation;  ρ=ρ+ε.*δρ)[1].total,0)
end

#première fonction pour calculer l'erreur en utilisant le differentiel

function erreur_energy(basis,scfres,basis_gamma)
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
    E=scfres.energies.total
    dE_schur=dE(basis_gamma, occ, ψr, res_schur, ρr)
    erreur= norm(E - dE_schur - E_ref)
end

# Affichage de l'erreur
#=S=zeros(21,7)
for j in 1:7
base_petite = PlaneWaveBasis(model; Ecut=5*(j-1)+5, kgrid)
scfres = self_consistent_field(base_petite; tol, callback=identity )
    for i in 1:21
        base_variable=PlaneWaveBasis(model; Ecut=(1+(1/10)*(i-1))*(5*(j-1)+5), kgrid)
        er=erreur_energy(base_petite,scfres,base_variable)
        S[i,j]=er
    end 
end
plot(xlabel="γ", ylabel="erreur relative", title="Variation de l'erreur de forces en fonction de γ", yscale=:log10)
for j in 1:7
    plot!(1:1/10:3, S[:, j], label="Ecut=$(5*(j-1)+5)")
end
plot!()
=#

# Deuxième fonction pour calculer l'erreur de l'énérgie en calculant l'énéergie de ψr-res_schur
function erreur_energy2(basis,scfres,basis_gamma)
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
    norm(E_approché- E_ref)
end

# Affichage de l'erreur en fonction de basis_gamma

S=zeros(21,7)
#plot(; xlabel="γ",ylabel="erreur relative",title="Variation de l'erreur en fonction de γ", yscale=:log10)
for j in 1:7
base_petite = PlaneWaveBasis(model; Ecut=5*(j-1)+5, kgrid)
scfres = self_consistent_field(base_petite; tol, callback=identity )
    for i in 1:21
        base_variable=PlaneWaveBasis(model; Ecut=(1+(1/10)*(i-1))*(5*(j-1)+5), kgrid)
        er=erreur_energy2(base_petite,scfres,base_variable)
        S[i,j]=er
    end 
end
plot(xlabel="γ", ylabel="erreur relative", title="Variation de l'erreur de forces en fonction de γ", yscale=:log10)
for j in 1:7
    plot!(1:1/10:3, S[:, j], label="Ecut=$(5*(j-1)+5)")
end
plot!()
