## Author: Ziyi Yin, ziyi.yin@gatech.edu
## Date: Sep 17, 2023

## Projection study - latent space scaling

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
using Statistics
using FNO4CO2
using InvertibleNetworks
using JSON
Random.seed!(2023)

matplotlib.use("agg")
include(srcdir("utils.jl"))

sim_name = "projection study"
exp_name = "fno-nf-proj-α-in-distribution"

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

obj(logK) = .5 * norm(S(T(prj(logK)), q)[1:length(tstep)*prod(n)]-state[1:length(tstep)*prod(n)])^2f0

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

## load FNO
device = cpu
net_path_FNO = datadir("trained-net", "trained-FNO.jld2")
net_dict_FNO = JLD2.jldopen(net_path_FNO, "r")
NN = net_dict_FNO["NN_save"] |> device;
AN = net_dict_FNO["AN"] |> device;
grid_ = gen_grid(net_dict_FNO["n"], net_dict_FNO["d"], net_dict_FNO["nt"], net_dict_FNO["dt"]) |> device;
Flux.testmode!(NN, true);

function SFNO(x)
    return clamp.(NN(perm_to_tensor(x, grid_, AN)), 0f0, 0.9f0);
end

logK0 = Float32.(logK0) |> device
@time y_init = SFNO(exp.(logK0)/Float32(md));
@time y_true = SFNO(exp.(Float32.(logK|>device))/Float32(md));

state_true = Saturations(state) |> device
println("FNO prediction error on true = ", norm(vec(y_true)-state_true)/norm(state_true))

ls = BackTracking(order=3, iterations=10)

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
G = G |> device;
G(zeros(Float32,n[1],n[end],1,1) |> device);
G1 = reverse(G);
z = zeros(Float32,prod(n)) |> device;

try
    global noiseLev = network_dict["noiseLev"]
catch e
    global noiseLev = network_dict["αmin"]
end

if exp_name == "fno-nf-proj-α-thickchannel"
    global K0 = 20f0 * ones(Float32, n[1], n[end]) |> device
    global K0[20:end, 20:40] .= 130f0
    global K0 += randn(Float32, size(K0)) * noiseLev * 120f0 |> device
elseif exp_name == "fno-nf-proj-α-in-distribution"
    global K0 = perm[:,:,end]
    global K0 += randn(Float32, size(K0)) * noiseLev * norm(K0, Inf)
    global K0 = K0 |> device
else
    global K0 = mean(perm, dims=3)[:,:,1] |> device;
    global K0 += noiseLev * randn(Float32, size(K0)) * 120f0 |> device
end
z = G1.inverse(reshape(K0,n[1],n[end],1,1));

function prjz(z::AbstractArray{T}; α=one(T)) where T
    znorm = norm(z)
    gaussian_norm = α * T(sqrt(length(z)))
    if znorm <= gaussian_norm
        return z
    else
        return z/znorm * gaussian_norm
    end
end

αlist = Vector{Float32}(range(0f0, stop=Float32(norm(z)/sqrt(length(z))), length=51))
l2list = zeros(Float32, length(αlist))
residual_list = zeros(Float32, length(αlist))
prj10130(x::AbstractArray{T}; upper=T(130), lower=T(10)) where T = max.(min.(x,T(upper)),T(lower))


PyPlot.rc("font", family="serif"); PyPlot.rc("xtick", labelsize=40); PyPlot.rc("ytick", labelsize=40)

for i = 1:length(αlist)
    α = αlist[i]
    zprj = prjz(z; α=α)
    Krec = prj10130(G1(vec(zprj))[:,:,1,1])
    l2list[i] = norm(Krec-K0)/norm(K0)

    global logK0 = prj(log.(Krec .* Float32(md)))
    co2true = Float32.(Saturations(S(T(Float64.(logK0|>cpu)), q))|>device)
    residual_list[i] = norm(vec(SFNO(exp.(logK0)/Float32(md)))-co2true)/norm(co2true)

    fig_name = @strdict n d α αlist Krec
    fig=figure(figsize=(20,12));imshow(Krec', aspect="auto", vmin=10f0, vmax=130f0);colorbar();title("projection ball size = $α * sqrt(length(z))", fontsize=40)
    safesave(joinpath(plotsdir(sim_name, exp_name), savename(fig_name; digits=6)*"_Krec.png"), fig);
    close(fig)

end


fig_name = @strdict n d αlist
extent = (0f0, n[1]*d[1], n[end]*d[end], 0f0)
fig = figure(figsize=(20,12));imshow(K0', vmin=10f0, vmax=130f0, extent=extent);
xlabel("X [m]", fontsize=40)
ylabel("Z [m]", fontsize=40)
cb = colorbar(pad=0.05, fraction=0.03)
cb[:set_label]("K [md]", fontsize=40);
for label in cb.ax.yaxis.get_ticklabels()
    label.set_rotation(0)
end
tight_layout()
safesave(joinpath(plotsdir(sim_name, exp_name), savename(fig_name; digits=6)*"_Kproj.png"), fig);
close(fig)

z = G1.inverse(reshape(K0,n[1],n[end],1,1));

fig_name = @strdict n d αlist
extent = (0f0, n[1]*d[1], n[end]*d[end], 0f0)
fig = figure(figsize=(20,12));imshow(z[:,:,1,1]', vmin=-3f0, vmax=3f0, cmap="seismic",extent=extent);
axis("off")
cb = colorbar(pad=0.05, fraction=0.03)
cb[:set_label]("z", fontsize=40);
for label in cb.ax.yaxis.get_ticklabels()
    label.set_rotation(0)
end
tight_layout()
safesave(joinpath(plotsdir(sim_name, exp_name), savename(fig_name; digits=6)*"_zproj.png"), fig);
close(fig)


fig=figure(figsize=(20,12));plot(αlist/maximum(αlist), l2list*100, linewidth=10, label="l2 misfit");
plot(αlist/maximum(αlist), residual_list*100, linewidth=10, label="FNO error");
axvline(x=1f0/maximum(αlist),color="red", linewidth=10, linestyle="--",label="white noise")
legend(fontsize=40);
xlabel(L"\alpha", fontsize=40);ylabel("[%]", fontsize=40)

safesave(joinpath(plotsdir(sim_name, exp_name), savename(fig_name; digits=6)*"_error.png"), fig);
close(fig)
