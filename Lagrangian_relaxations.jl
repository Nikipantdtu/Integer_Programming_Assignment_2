using JuMP
using GLPK
using Random
using Plots
using Printf

# Define data
# v_i: volumes of containers
v = [400, 450, 520, 330, 400, 350, 250, 300, 280, 310, 340, 290, 275, 180, 310]

# a_i: initial amount of liquid in containers 
a = [290, 240, 210, 300, 175, 190, 95, 190, 210, 80, 115, 95, 260, 140, 210]

n = length(v)  # number of containers
    
modelFull = Model(GLPK.Optimizer)
modelRAss = Model(GLPK.Optimizer)
modelRKS =  Model(GLPK.Optimizer)

@variable(modelFull, y[1:n], Bin)
@variable(modelFull, x[i=1:n, j=1:n; i != j], Bin)
@objective(modelFull, Min, sum(y[j] for j in 1:n))  

@constraint(modelFull, capacity[j=1:n], a[j] * y[j] + sum(a[i] * x[i, j] for i in 1:n if i != j) <= v[j] * y[j]) # Capacity constraints
@constraint(modelFull, assignment[i=1:n], y[i] + sum(x[i, j] for j in 1:n if j != i) == 1)  # Assignment constraints

optimize!(modelFull)
println("Optimal Objective Value: $(objective_value(modelFull))")
relax_integrality(modelFull)
optimize!(modelFull)
println("Optimal Value LP Relaxation: $(round(objective_value(modelFull), digits=2))")

lp_relax = objective_value(modelFull)   

# integer optimum from modelFull
opt_int = objective_value(modelFull)


## Relaxed Assignment Constraints (geom, dim, polyak all together)

# Decision variables (relaxed to [0,1] instead of Bin for the subproblem)
@variable(modelRAss, 0 <= yRAss[1:n] <= 1)
@variable(modelRAss, 0 <= xRAss[i=1:n, j=1:n] <= 1)

# No self-transfer
@constraint(modelRAss, [i=1:n], xRAss[i,i] == 0)

# Keep capacity constraints:
@constraint(modelRAss, capRAss[j=1:n],
    a[j] * yRAss[j] + sum(a[i] * xRAss[i,j] for i=1:n if i != j) <= v[j] * yRAss[j]
)

let

    maxK = 100

    # storage for each step rule
    LBAss_geom    = Float64[]
    LBAss_dim     = Float64[]
    LBAss_polyak  = Float64[]

    maxLB_geom    = -1.0e9
    maxLB_dim     = -1.0e9
    maxLB_polyak  = -1.0e9

    #################################
    # 1) GEOMETRIC STEP
    #################################
    u = zeros(n)             # multipliers for assignment cons (free sign)
    mu1 = 1
    rho = 0.9

    for k = 1:maxK
        # Lagrangian objective
        @objective(modelRAss, Min,
            sum(yRAss[j] for j=1:n)
            + sum(u[i] * (yRAss[i] + sum(xRAss[i,j] for j=1:n if j!=i) - 1 ) for i=1:n)
        )

        optimize!(modelRAss)
        objVal = objective_value(modelRAss)

        push!(LBAss_geom, objVal)
        if objVal > maxLB_geom + 1e-7
            maxLB_geom = objVal
        end

        # subgradient (assignment violation)
        gamma = [
            value(yRAss[i]) + sum(value(xRAss[i,j]) for j=1:n if j!=i) - 1
            for i=1:n
        ]

        # geometric step size
        muk = mu1 * (rho)^k

        # update multipliers
        for i=1:n
            u[i] = u[i] + muk * gamma[i]
        end
    end

    println("LB from Relaxation of Assignment constraint (geom): $(round(maxLB_geom, digits=4))")


    #################################
    # 2) DIMINISHING STEP
    #################################
    u .= 0.0                # reset multipliers
    for k = 1:maxK
        @objective(modelRAss, Min,
            sum(yRAss[j] for j=1:n)
            + sum(u[i] * (yRAss[i] + sum(xRAss[i,j] for j=1:n if j!=i) - 1 ) for i=1:n)
        )

        optimize!(modelRAss)
        objVal = objective_value(modelRAss)

        push!(LBAss_dim, objVal)
        if objVal > maxLB_dim + 1e-7
            maxLB_dim = objVal
        end

        gamma = [
            value(yRAss[i]) + sum(value(xRAss[i,j]) for j=1:n if j!=i) - 1
            for i=1:n
        ]

        muk = mu1 / k   # diminishing

        for i=1:n
            u[i] = u[i] + muk * gamma[i]
        end
    end

    println("LB from Relaxation of Assignment constraint (dim): $(round(maxLB_dim, digits=4))")


    #################################
    # 3) POLYAK STEP
    #################################
    u .= 0.0                      # reset multipliers again
    theta = 1.0                   # Polyak tuning parameter (0 < theta <= 2 typically)
    best_primal = opt_int         # incumbent integer solution cost (upper bound)

    for k = 1:maxK
        @objective(modelRAss, Min,
            sum(yRAss[j] for j=1:n)
            + sum(u[i] * (yRAss[i] + sum(xRAss[i,j] for j=1:n if j!=i) - 1 ) for i=1:n)
        )

        optimize!(modelRAss)
        objVal = objective_value(modelRAss)

        push!(LBAss_polyak, objVal)
        if objVal > maxLB_polyak + 1e-7
            maxLB_polyak = objVal
        end

        gamma = [
            value(yRAss[i]) + sum(value(xRAss[i,j]) for j=1:n if j!=i) - 1
            for i=1:n
        ]

        norm2_gamma = sum(gamma[i]^2 for i=1:n)

        if norm2_gamma == 0.0
            # subgradient is zero -> stationary for these multipliers
            break
        end

        stepsize_num = best_primal - objVal
        alpha_k = theta * max(0.0, stepsize_num) / norm2_gamma

        for i=1:n
            u[i] = u[i] + alpha_k * gamma[i]
        end
    end

    println("LB from Relaxation of Assignment constraint (polyak): $(round(maxLB_polyak, digits=4))")


    #################################
    # Plot all three + LP line
    #################################

    iters = 1:maxK
    # note: polyak loop may have broken early, so build a polyak_iters that matches
    polyak_iters = 1:length(LBAss_polyak)

    pRAss_all = plot(
        iters,
        LBAss_geom,
        label = "geom step",
        xlabel = "iteration",
        ylabel = "Lagrangian dual bound (LB value)",
        title = "Assignment Relaxation",
        seriestype = :path,
        marker = :circle,
        markersize = 4,
        linewidth = 2,
        legend = :bottomright,
    )

    plot!(
        iters,
        LBAss_dim,
        label = "diminishing step",
        seriestype = :path,
        marker = :square,
        markersize = 4,
        linewidth = 2,
    )

    plot!(
        polyak_iters,
        LBAss_polyak,
        label = "polyak step",
        seriestype = :path,
        marker = :diamond,
        markersize = 4,
        linewidth = 2,
    )

    # Add horizontal LP relaxation line (thin red dashed)
    plot!(
        iters,
        fill(lp_relax, maxK),
        label = @sprintf("LP relaxation (%.2f)", lp_relax),
        color = :red,
        linestyle = :dash,
        linewidth = 1.5,
    )

    display(pRAss_all)
    savefig(pRAss_all, joinpath(homedir(), "Desktop", "assignment_relaxation_plot.png"))

end # let



## Relaxed Capacity Constraints (geom, dim, polyak all together)

# Decision variables (relaxed [0,1])
@variable(modelRKS, 0 <= yRKS[1:n] <= 1)
@variable(modelRKS, 0 <= xRKS[i=1:n, j=1:n] <= 1)

# No self-transfer
@constraint(modelRKS, [i=1:n], xRKS[i,i] == 0)

# Keep the assignment constraints:
@constraint(modelRKS, assignRKS[i=1:n],
    yRKS[i] + sum(xRKS[i,j] for j=1:n if j != i) == 1
)

let
    maxK = 100

    # store LB trajectories
    LBKSCap_geom    = Float64[]
    LBKSCap_dim     = Float64[]
    LBKSCap_polyak  = Float64[]

    # best-so-far (true lower bound candidates)
    maxLBcap_geom    = -1.0e9
    maxLBcap_dim     = -1.0e9
    maxLBcap_polyak  = -1.0e9

    #################################
    # common helpers
    #################################
    function set_cap_lagrangian_objective!(beta_vec)
        @objective(modelRKS, Min,
            sum(yRKS[j] for j=1:n)
            +
            sum(
                beta_vec[j] * (
                    a[j] * yRKS[j] +
                    sum(a[i] * xRKS[i,j] for i=1:n if i != j) -
                    v[j] * yRKS[j]
                )
                for j=1:n
            )
        )
    end

    # Computes gammaCap (capacity violations) after optimize!(modelRKS)
    function capacity_violations()
        [
            (
                a[j] * value(yRKS[j])
                + sum(a[i] * value(xRKS[i,j]) for i=1:n if i != j)
                - v[j] * value(yRKS[j])
            )
            for j=1:n
        ]
    end

    # optional upper cap on beta to stop craziness
    cap_val = 1.0e3  # you can lower to 1.0e2 if you want even calmer multipliers

    #################################
    # 1) GEOMETRIC STEP
    #################################
    beta = zeros(n)      # β ≥ 0 multipliers for capacity
    mu1  = 0.000001          # <- damped base step size (was 1.0)
    rho  = 0.9           # geometric decay

    for k = 1:maxK
        # set objective with current beta
        set_cap_lagrangian_objective!(beta)

        optimize!(modelRKS)
        objVal = objective_value(modelRKS)

        push!(LBKSCap_geom, objVal)
        if objVal > maxLBcap_geom + 1e-7
            maxLBcap_geom = objVal
        end

        gammaCap = capacity_violations()

        muk = mu1 * (rho)^k

        # projected update beta[j] >= 0, and cap to avoid explosion
        for j=1:n
            beta[j] = max(0.0, beta[j] + muk * gammaCap[j])
            beta[j] = min(cap_val, beta[j])   # <- cap (optional but helps stability)
        end
    end

    @printf("LB from Relaxation of Capacity constraint (geom): %.10e\n", maxLBcap_geom)


    #################################
    # 2) DIMINISHING STEP
    #################################
    beta .= 0.0
    mu1  = 0.000001          # same damped scale

    for k = 1:maxK
        set_cap_lagrangian_objective!(beta)

        optimize!(modelRKS)
        objVal = objective_value(modelRKS)

        push!(LBKSCap_dim, objVal)
        if objVal > maxLBcap_dim + 1e-7
            maxLBcap_dim = objVal
        end

        gammaCap = capacity_violations()

        muk = mu1 / k   # diminishing

        for j=1:n
            beta[j] = max(0.0, beta[j] + muk * gammaCap[j])
            beta[j] = min(cap_val, beta[j])   # cap again
        end
    end

    @printf("LB from Relaxation of Capacity constraint (dim): %.10e\n", maxLBcap_dim)


    #################################
    # 3) POLYAK STEP
    #################################
    beta .= 0.0
    theta        = 1.0           # Polyak tuning parameter
    best_primal  = opt_int       # incumbent feasible integer solution cost

    for k = 1:maxK
        set_cap_lagrangian_objective!(beta)

        optimize!(modelRKS)
        objVal = objective_value(modelRKS)

        push!(LBKSCap_polyak, objVal)
        if objVal > maxLBcap_polyak + 1e-7
            maxLBcap_polyak = objVal
        end

        gammaCap = capacity_violations()
        norm2_gammaCap = sum(gammaCap[j]^2 for j=1:n)

        if norm2_gammaCap == 0.0
            # already capacity-feasible with this beta -> stationary
            break
        end

        stepsize_num = best_primal - objVal
        alpha_k = theta * max(0.0, stepsize_num) / norm2_gammaCap

        for j=1:n
            beta[j] = max(0.0, beta[j] + alpha_k * gammaCap[j])
            beta[j] = min(cap_val, beta[j])   # cap to control blow-up
        end
    end

    @printf("LB from Relaxation of Capacity constraint (polyak): %.10e\n", maxLBcap_polyak)


    #################################
    # Plot all three + LP relaxation
    #################################
    iters_all   = 1:maxK
    iters_poly  = 1:length(LBKSCap_polyak)  # Polyak may break early

    pRKS_all = plot(
        iters_all,
        LBKSCap_geom,
        label = "geom step",
        xlabel = "iteration",
        ylabel = "Lagrangian dual bound (LB value)",
        title = "Capacity Relaxation",
        seriestype = :path,
        marker = :circle,
        markersize = 4,
        linewidth = 2,
        legend = :bottomright,
    )

    plot!(
        iters_all,
        LBKSCap_dim,
        label = "diminishing step",
        seriestype = :path,
        marker = :square,
        markersize = 4,
        linewidth = 2,
    )

    plot!(
        iters_poly,
        LBKSCap_polyak,
        label = "polyak step",
        seriestype = :path,
        marker = :diamond,
        markersize = 4,
        linewidth = 2,
    )

    # Add horizontal LP relaxation line (thin red dashed)
    plot!(
        iters_all,
        fill(lp_relax, maxK),
        label = @sprintf("LP relaxation (%.2f)", lp_relax),
        color = :red,
        linestyle = :dash,
        linewidth = 1.5,
    )

    display(pRKS_all)
    savefig(pRKS_all, joinpath(homedir(), "Desktop", "capacity_relaxation_plot.png"))


end # let
