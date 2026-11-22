# Stochastic simulation (Gillespie's algorithm) of the reaction
#     A --(k)--> B,   k = 0.1, Na = 100, Nb = 0
# from t = 0 to t = 100 ms.
#
# The simulation is run 1000 times in parallel using Distributed.jl, and the
# mean number of molecules is taken at each time. This lets us compare the
# average of the stochastic runs against the exact solution
#     A(t) = Na * exp(-k t),   B(t) = Na * (1 - exp(-k t)).

using Distributed
if nworkers() == 1
    # Start 4 worker processes so simulations can run in parallel
    addprocs(4)
end

# Run code on the main process and each worker
@everywhere begin
    function gillespie(; k=0.1, Na0=100, T=100.0)
        Na, Nb, t = Na0, 0, 0.0
        ts = Float64[0.0]
        mol_A = Int[Na]
        mol_B = Int[Nb]

        while Na > 0
            tau = -log(rand()) / (k * Na)
            t += tau
            t > T && break
            Na -= 1
            Nb += 1
            push!(ts, t)
            push!(mol_A, Na)
            push!(mol_B, Nb)
        end

        return ts, mol_A, mol_B
    end

    # Run one simulation and sample it at the grid times so runs can be averaged
    function run_on_grid(grid; kwargs...)
        ts, A, B = gillespie(; kwargs...)
        idx = searchsortedlast.(Ref(ts), grid)
        return A[idx], B[idx]
    end
end

using Plots

k = 0.1
Na0 = 100
T = 100.0
nruns = 1000
grid = range(0, T, length=500)

println("Running $nruns simulations on $(nworkers()) workers...")
elapsed = @elapsed results = pmap(_ -> run_on_grid(grid; k=k, Na0=Na0, T=T), 1:nruns)
println("Simulations finished in $(round(elapsed, digits=2)) s")

A_all = reduce(hcat, first.(results))
B_all = reduce(hcat, last.(results))
A_mean = vec(sum(A_all, dims=2)) ./ nruns
B_mean = vec(sum(B_all, dims=2)) ./ nruns

println("Mean A at t = $T ms: ", A_mean[end], "   (exact: ", Na0 * exp(-k * T), ")")
println("Mean B at t = $T ms: ", B_mean[end], "   (exact: ", Na0 * (1 - exp(-k * T)), ")")

nshow = min(30, nruns)
plot(grid, A_all[:, 1:nshow], color=1, alpha=0.15, label="")
plot!(grid, B_all[:, 1:nshow], color=2, alpha=0.15, label="")
plot!(grid, A_mean, color=1, lw=2, label="A (mean of $nruns runs)")
plot!(grid, B_mean, color=2, lw=2, label="B (mean of $nruns runs)")
plot!(grid, Na0 .* exp.(-k .* grid), ls=:dash, color=:black, label="exact")
plot!(grid, Na0 .* (1 .- exp.(-k .* grid)), ls=:dash, color=:black, label="")
plot!(legend=:right)
xlabel!("Time (ms)")
ylabel!("Number of Molecules")
title!("A → B, k = $k")

savefig("A_to_B.svg")