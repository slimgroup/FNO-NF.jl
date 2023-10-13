## A 2D example

using DrWatson
@quickactivate "FNO-NF"

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
using Random
Random.seed!(2023)

include(srcdir("dummy_src_file.jl"))

sim_name = "gen-train"
exp_name = "flow-channel"

mkpath(datadir())
mkpath(plotsdir())

# Define raw data directory
mkpath(datadir("training-data"))
perm_path = datadir("training-data", "cons=1e-5_delta=25_num_sample=10000_theta0=5.jld2")

# Download the dataset into the data directory if it does not exist
if ~isfile(perm_path)
    run(`wget https://www.dropbox.com/s/xy36bvoz6iqau60/'
        'cons=1e-5_delta=25_num_sample=10000_theta0=5.jld2 -q -O $perm_path`)
end

JLD2.@load "../data/training-data/cons=1e-5_delta=25_num_sample=10000_theta0=5.jld2"

## grid size
n = (64, 1, 64)
d = (15.0, 10.0, 15.0)

## permeability
K = md * perm[:,:,1]
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
#q = jutulSource(irate, inj_loc)

## set up modeling operator
S = jutulModeling(model, tstep)

## simulation
Trans = KtoTrans(CartesianMesh(model), K1to3(K))
@time result = S(log.(Trans), q)

## plotting
fig=figure(figsize=(20,12));
subplot(1,2,1);
imshow(reshape(Saturations(result.states[end]), n[1], n[end])', vmin=0, vmax=1); colorbar(); title("saturation")
subplot(1,2,2);
imshow(reshape(Pressure(result.states[end]), n[1], n[end])', vmin=minimum(Pressure(result.states[end])), vmax=maximum(Pressure(result.states[end]))); colorbar(); title("pressure")

nsample = 2000
nt = length(tstep)
Ks = zeros(n[1], n[end], nsample);
conc = zeros(n[1], n[end], nt, nsample);
pres = zeros(n[1], n[end], nt, nsample);

for i = 1:nsample

    Base.flush(Base.stdout)

    println("sample $(i)")
    Ks[:,:,i] = perm[:,:,i]
    K = md * Ks[:,:,i]
    Trans = KtoTrans(CartesianMesh(model), K1to3(K))
    @time state = S(log.(Trans), q)
    conc[:,:,:,i] = reshape(Saturations(state), n[1], n[end], nt)
    pres[:,:,:,i] = reshape(Pressure(state), n[1], n[end], nt)
end

save_dict = @strdict n d nsample ϕ tstep inj_loc prod_loc irate Ks conc pres
@tagsave(
    joinpath(datadir(sim_name, exp_name), savename(save_dict, "jld2"; digits=6)),
    save_dict;
    safe=true
)
