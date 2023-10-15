## Author: Ziyi Yin, ziyi.yin@gatech.edu
## Date: Sep 17, 2023

## Permeability inversion
## Observed data: well
## Methods: constrained optimization with numerical solver

using DrWatson
@quickactivate "FNO-NF"
using Pkg; Pkg.add(url="https://github.com/slimgroup/FNO4CO2/", rev="v1.1.4")
using Pkg; Pkg.instantiate();

nthreads = try
    # Slurm
    parse(Int, ENV["SLURM_CPUS_ON_NODE"])
catch e
    # Desktop
    Sys.CPU_THREADS
end
using LinearAlgebra
BLAS.set_num_threads(nthreads)

using JutulDarcyRules
using PyPlot
using JLD2
using Flux
using Random
using LineSearches
using InvertibleNetworks
using FNO4CO2
using Statistics
using JOLI
using JSON
Random.seed!(2023)

matplotlib.use("agg")
include(srcdir("utils.jl"))

sim_name = "flow-inversion"
exp_name = "jutul-nf"

JLD2.@load datadir("examples", "K.jld2") K

mkpath(datadir())
mkpath(plotsdir())

## grid size
n = (64, 1, 64)
d = (15.0, 10.0, 15.0)

## permeability
K = md * K
ϕ = 0.25 * ones(n)
model = jutulModel(n, d, vec(ϕ), K1to3(K))

## simulation time steppings
tstep = 100 * ones(8)
tot_time = sum(tstep)

## injection & production
inj_loc = (3, 1, 32) .* d
prod_loc = (62, 1, 32) .* d
irate = 5e-3
q = jutulSource(irate, [inj_loc, prod_loc])

## set up modeling operator
S = jutulModeling(model, tstep)

## simulation
logK = log.(K)
mesh = CartesianMesh(model)
T(x) = log.(KtoTrans(mesh, K1to3(exp.(x))))
@time state = S(T(logK), q)

prj(x::AbstractArray{T}; upper=T(log(130*md)), lower=T(log(10*md))) where T = max.(min.(x,T(upper)),T(lower))

# Main loop
niterations = 100
fhistory = zeros(niterations)

obs_loc = zeros(Float32, n[1], n[end], length(tstep))
subsamp = 3
obs_loc[Int.(round.(range(1, stop=n[1], length=subsamp))),:,:] .= 1f0
obs_loc = vec(obs_loc)

# Define raw data directory
mkpath(datadir("gen-train","flow-channel"))
perm_path = joinpath(datadir("gen-train","flow-channel"), "irate=0.005_nsample=2000.jld2")

# Download the dataset into the data directory if it does not exist
if ~isfile(perm_path)
    run(`wget https://www.dropbox.com/s/8jb5g4rmamigoqf/'
        'irate=0.005_nsample=2000.jld2 -q -O $perm_path`)
end

dict_data = JLD2.jldopen(perm_path, "r")
perm = Float32.(dict_data["Ks"]);

K0 = mean(perm, dims=3)[:,:,1] * md
logK0 = log.(K0)
@time state_init = S(T(logK0), q)

# load the NF network
net_path = datadir("trained-net", "trained-NF.jld2")
network_dict = JLD2.jldopen(net_path, "r");

G = NetworkMultiScaleHINT(1, network_dict["n_hidden"], network_dict["L"], network_dict["K"];
                               split_scales=true, max_recursion=network_dict["max_recursion"], p2=0, k2=1, activation=SigmoidLayer(low=0.5f0,high=1.0f0), logdet=false);
P_curr = get_params(G);
for j=1:length(P_curr)
    P_curr[j].data = network_dict["Params"][j].data;
end

# forward to set up splitting, take the reverse for Asim formulation
device = cpu
G = G |> device;
G(zeros(Float32,n[1],n[end],1,1) |> device);
G1 = reverse(G);
z = zeros(Float32,prod(n)) |> device;

try
    global noiseLev = network_dict["noiseLev"]
catch e
    global noiseLev = network_dict["αmin"]
end

K0 = mean(perm, dims=3)[:,:,1] |> device;
K0 += noiseLev * randn(Float32, size(K0)) * 120f0 |> device
z = vec(G1.inverse(reshape(K0,n[1],n[end],1,1)));

obj(z) = .5 * norm(obs_loc .* (S(T(Float64.(log.(box_K(G1(z)[:,:,1,1]) .* Float32(md)))), q)[1:length(tstep)*prod(n)]-state[1:length(tstep)*prod(n)]))^2f0

τ = 0.4f0
β = 1.2f0
τinit = deepcopy(τ)
τend = 1.2f0
z = prjz(z; α=τ)

ls = BackTracking(c_1=1f-4,iterations=50,maxstep=Inf32,order=3,ρ_hi=5f-1,ρ_lo=1f-1)

for j=1:niterations

    global logK0 = log.(box_K(G1(z)[:,:,1,1]) .* Float32(md))
    @time fval, gs = Flux.withgradient(() -> obj(z), Flux.params(z))
    g = gs[z]
    p = -g/norm(g, Inf)
  
    println("Inversion iteration no: ",j,"; function value: ",fval)
    fhistory[j] = fval

    # linesearch
    function f_(α)
        misfit = obj(Float32.(prjz(z .+ α * p; α=τ)))
        @show α, misfit
        return misfit
    end

    step, fval = ls(f_, 5f-1, fval, dot(g, p))

    # Update model and bound projection
    global z = Float32.(prjz(z .+ step .* p; α=τ))
    global logK0 = box_logK(log.(box_K(G1(z)[:,:,1,1]) .* Float32(md)))

    fig_name = @strdict j n d ϕ logK0 tstep irate niterations inj_loc β τinit τend subsamp

    ### plotting
    fig=figure(figsize=(20,12));
    subplot(1,3,1);
    imshow(exp.(logK)'./md, vmin=minimum(exp.(logK))./md, vmax=maximum(exp.(logK)./md)); colorbar(); title("true permeability")
    subplot(1,3,2);
    imshow(exp.(logK0)'./md, vmin=minimum(exp.(logK))./md, vmax=maximum(exp.(logK)./md)); colorbar(); title("inverted permeability")
    subplot(1,3,3);
    imshow(abs.(exp.(logK)'./md.-exp.(logK0)'./md), vmin=minimum(exp.(logK)), vmax=maximum(exp.(logK)./md)); colorbar(); title("diff")
    suptitle("Flow Inversion at iter $j")
    tight_layout()
    safesave(joinpath(plotsdir(sim_name, exp_name), savename(fig_name; digits=6)*"_diff.png"), fig);
    close(fig)

    state_predict = S(T(Float64.(logK0)), q)

    ## data fitting
    fig = figure(figsize=(20,12));
    for i = 1:5
        subplot(4,5,i);
        imshow(reshape(Saturations(state_init.states[3+i]), n[1], n[end])', vmin=0, vmax=0.9); colorbar();
        title("initial prediction at snapshot $(3+i)")
        subplot(4,5,i+5);
        imshow(reshape(Saturations(state.states[3+i]), n[1], n[end])', vmin=0, vmax=0.9); colorbar();
        title("true at snapshot $(3+i)")
        subplot(4,5,i+10);
        imshow(reshape(Saturations(state_predict.states[3+i]), n[1], n[end])', vmin=0, vmax=0.9); colorbar();
        title("predict at snapshot $(3+i)")
        subplot(4,5,i+15);
        imshow(5*abs.(reshape(Saturations(state.states[3+i]), n[1], n[end])'-reshape(Saturations(state_predict.states[3+i]), n[1], n[end])'), vmin=0, vmax=0.9); colorbar();
        title("5X diff at snapshot $(3+i)")
    end
    suptitle("Flow Inversion at iter $j")
    tight_layout()
    safesave(joinpath(plotsdir(sim_name, exp_name), savename(fig_name; digits=6)*"_co2.png"), fig);
    close(fig)

    ## loss
    fig = figure(figsize=(20,12));
    plot(fhistory[1:j]);title("loss=$(fhistory[j])");
    suptitle("Flow Inversion at iter $j")
    tight_layout()
    safesave(joinpath(plotsdir(sim_name, exp_name), savename(fig_name; digits=6)*"_loss.png"), fig);
    close(fig)

end

save_dict = @strdict sim_name exp_name subsamp niterations logK0 fhistory
@tagsave(
    joinpath(datadir(sim_name, exp_name), savename(save_dict, "jld2"; digits=6)),
        save_dict;
        safe=true
)
