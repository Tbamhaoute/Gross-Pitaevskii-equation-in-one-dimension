using JLD2, Plots
using LaTeXStrings
# Charger les données depuis le fichier JLD2
data = JLD2.load("data_erreur.jld2")    #changer par le nom du fichier jld2
data_courbes_erreur = data["data_courbes_erreur"]
data_courbes_gamma = data["data_courbes_gamma"]
gamma_list = data["gamma_list"]
f_liste = data["f_liste"]
g_liste = data["g_liste"]

# Extraire les paires (clé, valeur numérique d'Ecut)
pairs = [(key, parse(Int, split(key, "=")[2])) for key in keys(data_courbes_erreur)]

# Trier les paires en fonction de la valeur numérique d'Ecut
sorted_pairs = sort(pairs, by=x -> x[2])

# Graphique principal
plot(xlabel="γ", ylabel="Erreur d'énergie", title="Variation de l'erreur d'énergie en fonction de γ")

# Tracé des courbes d'erreur pour chaque valeur de Ecut trié
for (key, _) in sorted_pairs
    value = data_courbes_erreur[key]
    Ecut_list = value[:Ecut_list]
    err = value[:err]
    
    # Trier les valeurs d'Ecut_list et err en fonction de Ecut_list
    sorted_idx = sortperm(Ecut_list)
    Ecut_list_sorted = Ecut_list[sorted_idx]
    err_sorted = err[sorted_idx]

    # Tracer les courbes avec les valeurs triées
    plot!(Ecut_list_sorted, err_sorted, yscale=:log10, label=key)  # Ajouter les labels pour chaque courbe
end

# Affichage final
p1=plot!()

pairs = [(key, parse(Int, split(key, "=")[2])) for key in keys(data_courbes_gamma)]

# Trier les paires en fonction de la valeur numérique d'Ecut
sorted_pairs = sort(pairs, by=x -> x[2])

# Graphique principal
plot(xlabel="γ", 
     ylabel=L"||M_{22} e_2 - (\Omega + K)_{22} e_2 ||", 
     title="Variation de " * L"||M_{22} e_2 - (\Omega + K)_{22} e_2 ||" * " en fonction de γ", 
     legend=:topright)

# Tracé des courbes triées par Ecut
for (key, _) in sorted_pairs
    value = data_courbes_gamma[key]
    Ecut_list = value[:Ecut_list]
    gamm = value[:gamm]
    
    # Trier les valeurs d'Ecut_list et gamm en fonction de Ecut_list
    sorted_idx = sortperm(Ecut_list)
    Ecut_list_sorted = Ecut_list[sorted_idx]
    gamm_sorted = gamm[sorted_idx]

    # Tracer les courbes avec les valeurs triées
    plot!(Ecut_list_sorted, gamm_sorted, label=key)
end

# Ajouter les points de stagnation avec un scatter plot
p2=plot!(gamma_list, g_liste, seriestype=:scatter, marker=:circle, color=:red, label="Points de stagnation")
display([p1,p2])
