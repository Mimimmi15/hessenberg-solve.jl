using LinearAlgebra

# 1. Modifiez la fonction suivante pour qu'elle renvoie la solution x
#    du système triangulaire supérieur Rx = b.
#    Votre fonction ne doit modifier ni R ni b.
"""
Résout un système Hessenberg supérieur Hx = b par substitution arrière modifiée.

# Arguments
- `R`: Matrice Hessenberg supérieure (n×n)
- `b`: Vecteur du membre de droite (n×1)

# Returns
- `x`: Solution du système Hx = b
"""
function backsolve(R::UpperTriangular, b)
    x = similar(b)
    n = size(R, 1)
    # On parcourt les indices de n jusqu'à 1 (de bas en haut)
    for i = n:-1:1
        x[i] = b[i]
        # On soustrait les termes déjà calculés
        for j = (i+1):n
            x[i] -= R[i,j] * x[j]
        end
        # On divise par le terme diagonal
        x[i] /= R[i,i]
    end
    return x
end

# 2. Modifiez la fonction suivante pour qu'elle renvoie la solution x
#    du système Hessenberg supérieur Hx = b ou du problème aux moindres
#    carrés min ‖Hx - b‖ à l'aide de rotations ou
#    réflexions de Givens et d'une remontée triangulaire.
#    Votre fonction peut modifier H et b si nécessaire.
#    Il n'est pas nécessaire de garder les rotations en mémoire et la
#    fonction ne doit pas les renvoyer.
#    Seul le cas réel sera testé ; pas le cas complexe.

"""
Résout le problème aux moindres carrés min ||Hx - b|| pour une matrice Hessenberg supérieure
où H a plus de lignes que de colonnes.

# Arguments
- `H`: Matrice Hessenberg supérieure (m×n) avec m > n
- `b`: Vecteur du membre de droite (m×1)

# Returns
- `x`: Solution aux moindres carrés
"""
function hessenberg_solve(H::UpperHessenberg, b)
    m, n = size(H)

    for i in 1:(m - 1)
        a, e = H[i, i], H[i+1, i]
        r = sqrt(a^2 + e^2)
        c = a / r
        s = e / r
        # On applique la rotation de Givens à la ligne i et i+1 de H 
        for j in i:n
            z= H[i, j]
            H[i, j] = c * H[i, j] + s * H[i+1, j]
            H[i+1, j] = -s * z + c * H[i+1, j]
        end
         # On applique la rotation à b
        z =  b[i] 
        b[i] = c * b[i] + s * b[i+1]
        b[i+1] = -s * z + c * b[i+1]
    end

    R = UpperTriangular(H[1:n, 1:n])
    return backsolve(R, b[1:n])
end 


# vérification
using Test
for n ∈ (10, 20, 30)
    # square system
    A = rand(n, n)
    A[diagind(A)] .+= 1
    b = rand(n)
    R = UpperTriangular(A)
    x = backsolve(R, b)
    @test norm(R * x - b) ≤ sqrt(eps()) * norm(b)
    H = UpperHessenberg(A)
    x = hessenberg_solve(copy(H), copy(b))
    @test norm(H * x - b) ≤ sqrt(eps()) * norm(b)
    # slightly overdetermined least squares
    A = rand(n + 1, n)
    A[diagind(A)] .+= 1
    H = UpperHessenberg(A)
    b = rand(n + 1)
    x_ls = hessenberg_solve(copy(H), copy(b))
    x_qr = H \ b
    @test norm(x_ls - x_qr) ≤ sqrt(eps()) * norm(x_qr)
end
