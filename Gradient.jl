module Gradient
include fonction

function gradient_descent(Nb, alpha, beta, max_iterations, tolerance) #méhode de gradient descent à pas fixe
    dx = 1 / Nb
    h = construction(Nb)
    P_old = phi(h)
    P_new = zeros(Nb,Nb)
    s = []
    erreur = []
    for k in 1:max_iterations
        H = h + (alpha / dx) * diagm(diag(P_old))
        P_new = P_old - beta * projection(P_old,H)
        P_new=Symmetric(P_new)
        P_new = retraction(P_new)
        push!(s,k)
        push!(erreur, norm(P_new - P_old))
        # Vérification de la convergence
        if norm(P_new - P_old) < tolerance
            println("Convergence atteinte après $k itérations.")
                 return P_new, s, erreur
            break
        end
        P_old=P_new
    end
return P_new, s, erreur
end

function gradient_descent_variable(Nb, alpha, max_iterations, tolerance) #méthode de gradient descent à pas variable avec la trace
    dx = 1 / Nb
    h = construction(Nb)
    P_old = phi(h)
    P_new = zeros(Nb,Nb)
    s = []
    erreur = []
    for k in 1:max_iterations
        H = h + (alpha / dx) * diagm(diag(P_old))
        proj=projection(P_old,H)
        beta=(tr(h*P_old*proj)+(alpha/(2*dx))*sum(P_old[i,i]*proj[i,i] for i in 1:Nb))/(tr(h*proj^2)+(alpha/(2*dx))*sum(proj[i,i]^2 for i in 1:Nb))
        P_new = P_old - beta * projection(P_old,H)
        P_new=Symmetric(P_new)
        P_new = retraction(P_new)
        push!(s,k)
        push!(erreur, norm(P_new - P_old))
        # Vérification de la convergence
        if norm(P_new - P_old) < tolerance
         println("Convergence atteinte après $k itérations.")
                 return P_new, s, erreur
            break
        end
        P_old=P_new
    end
return P_new, s, erreur
end

function gradient_descent_random(Nb, alpha, max_iterations, tolerance)  #multiplication du beta variable par une loi uniforme entre [0,2]
    dx = 1 / Nb
    h = construction(Nb)
    P_old = phi(h)
    P_new = zeros(Nb,Nb)
    s = []
    erreur = []
    for k in 1:max_iterations
        H = h + (alpha / dx) * diagm(diag(P_old))
        proj=projection(P_old,H)
        beta=2*rand()*(tr(h*P_old*proj)+(alpha/(2*dx))*sum(P_old[i,i]*proj[i,i] for i in 1:Nb))/(tr(h*proj^2)+(alpha/(2*dx))*sum(proj[i,i]^2 for i in 1:Nb))
        P_new = P_old - beta * projection(P_old,H)
        P_new=Symmetric(P_new)
        P_new = retraction(P_new)
        push!(s,k)
        push!(erreur, norm(P_new - P_old))
        # Vérification de la convergence
        if norm(P_new - P_old) < tolerance
         #println("Convergence atteinte après $k itérations.")
                 return P_new, s, erreur
            break
        end
        P_old=P_new
    end
return P_new, s, erreur
end

function gradient_barzilai(Nb, alpha, beta, max_iter, tolerance)  #méthode de barzilai-borwein pour le calcule du pas
    dx = 1 / Nb
    h = construction(Nb)
    P_old = phi(h)
    s = []
    erreur = []
    beta_k=[beta]
    for k in 1:max_iter
        H = h + (alpha / dx) * diagm(diag(P_old))
        P_new = phi(H)
        P_new = P_old - beta * projection(P_old,H)
        P_new=Symmetric(P_new)
        P_new=retraction(P_new)
        beta=(norm(P_new-P_old)^2)/tr((P_new-P_old)*(projection(P_new,h + (alpha / dx) * diagm(diag(P_new)))-projection(P_old,H))')
        push!(beta_k,beta)
        push!(s, k)
        push!(erreur, norm(P_new - P_old))

        # Vérification de la convergence
        if norm(P_new - P_old) < tolerance
            println("Convergence atteinte après $k itérations.")
            return P_old, s, erreur,beta_k
        end
       
        P_old = P_new
    end

    println("Nombre maximum d'itérations atteint.")
    return P_old, s, erreur,beta_k
end

end #module
