# Consider the following chemical reaction:

# A --(k)--> B, where k = 0.1, Na = 100, Nb = 0

# We will stochastically model the reaction from time t=0 ms to t = 1000 ms

using Plots

function gillespie(; k=0.1, Na0=100, T=1000.0)
    Na = Na0
    Nb = 0
    t  = 0.0

    ts    = Float64[0.0]
    mol_A = Int[Na]
    mol_B = Int[Nb]

    while Na > 0
        w   = k * Na
        tau = -log(rand()) / w
        t  += tau

        if t > T
            break
        end

        Na -= 1
        Nb += 1
        push!(ts, t)
        push!(mol_A, Na)
        push!(mol_B, Nb)
    end

    return ts, mol_A, mol_B
end

k, T = 0.1, 1000.0
ts, mol_A, mol_B = gillespie(k=k, T=T)

t_plot = [ts; T]
A_plot = [mol_A; mol_A[end]]
B_plot = [mol_B; mol_B[end]]

t_exact = range(0, T, length=1000)
A_exact = 100 .* exp.(-k .* t_exact)
B_exact = 100 .* (1 .- exp.(-k .* t_exact))
 
plot(t_plot, A_plot, seriestype=:steppost, label="A (stochastic)", lw=2,
     xlims=(0, 100))
plot!(t_plot, B_plot, seriestype=:steppost, label="B (stochastic)", lw=2)
plot!(t_exact, A_exact, label="A (exact mean)", ls=:dash, color=:black)
plot!(t_exact, B_exact, label="B (exact mean)", ls=:dash, color=:black)
xlabel!("Time (ms)")
ylabel!("Number of Molecules")
title!("A → B, k = $k")
 
savefig("A_to_B.svg")