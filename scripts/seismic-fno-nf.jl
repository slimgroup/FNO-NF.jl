## Author: Ziyi Yin, ziyi.yin@gatech.edu
## Date: Sep 17, 2023

## Permeability inversion
## Observed data: seismic
## Methods: constrained optimization with surrogates

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
using JUDI
using JSON
using InvertibleNetworks
Random.seed!(2023)

matplotlib.use("agg")
include(srcdir("utils.jl"))

sim_name = "end2end-inversion"
exp_name = "fno-nf"

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
f = jutulSource(irate, [inj_loc, prod_loc])

## set up modeling operator
S = jutulModeling(model, tstep)

## simulation
logK = log.(K)
mesh = CartesianMesh(model)
T(x) = log.(KtoTrans(mesh, K1to3(exp.(x))))
@time state = S(T(logK), f)

prj(x::AbstractArray{T}; upper=T(log(130*md)), lower=T(log(10*md))) where T = max.(min.(x,T(upper)),T(lower))

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

### observed states
nv = 6
survey_indices = 1:nv
O(state::AbstractArray) = [state[:,:,i] for i = 1:nv]
function O(state::jutulStates)
    full_his = Float32.(reshape(state[1:nv*prod(ns)], ns[1], ns[end], nv))
    return [full_his[:,:,i] for i = 1:nv]
end
O(state::AbstractVector) = Float32.(permutedims(reshape(state[1:length(tstep)*prod(n)], n[1], n[end], length(tstep)), [3,1,2])[survey_indices,:,:])
sw_true = O(state)

# set up rock physics
vp = 3500 * ones(Float32,n[1],n[end])     # p-wave
phi = 0.25f0 * ones(Float32,n[1],n[end])  # porosity
rho = 2200 * ones(Float32,n[1],n[end])    # density
R(c::AbstractArray{Float32,3}) = Patchy(c,vp,rho,phi)[1]
R(c::Vector{Matrix{Float32}}) = Patchy(c,vp,rho,phi)[1]
vps = R(sw_true)   # time-varying vp

## upsampling
upsample = 2
u(x::Vector{Matrix{Float32}}) = [repeat(x[i], inner=(upsample,upsample)) for i = 1:nv]
vpups = u(vps)

##### Wave equation
nw = (n[1], n[end]).*upsample
dw = (15f0/upsample, 15f0/upsample)        # discretization for wave equation
o = (0f0, 0f0)          # origin

nsrc = 32       # num of sources
nrec = 960      # num of receivers

models = [Model(nw, dw, o, (1f3 ./ vpups[i]).^2f0; nb = 80) for i = 1:nv]   # wave model

timeS = timeR = 750f0               # recording time
dtS = dtR = 1f0                     # recording time sampling rate
ntS = Int(floor(timeS/dtS))+1       # time samples
ntR = Int(floor(timeR/dtR))+1       # source time samples

# source locations -- half at the left hand side of the model, half on top
xsrc = convertToCell(vcat(range(dw[1],stop=dw[1],length=Int(nsrc/2)),range(dw[1],stop=(nw[1]-1)*dw[1],length=Int(nsrc/2))))
ysrc = convertToCell(range(0f0,stop=0f0,length=nsrc))
zsrc = convertToCell(vcat(range(dw[2],stop=(nw[2]-1)*dw[2],length=Int(nsrc/2)),range(10f0,stop=10f0,length=Int(nsrc/2))))

# receiver locations -- half at the right hand side of the model, half on top
xrec = vcat(range((nw[1]-1)*dw[1],stop=(nw[1]-1)*dw[1], length=Int(nrec/2)),range(dw[1],stop=(nw[1]-1)*dw[1],length=Int(nrec/2)))
yrec = 0f0
zrec = vcat(range(dw[2],stop=(nw[2]-1)*dw[2],length=Int(nrec/2)),range(10f0,stop=10f0,length=Int(nrec/2)))

# set up src/rec geometry
srcGeometry = Geometry(xsrc, ysrc, zsrc; dt=dtS, t=timeS)
recGeometry = Geometry(xrec, yrec, zrec; dt=dtR, t=timeR, nsrc=nsrc)

# set up source
f0 = 0.05f0     # kHz
wavelet = ricker_wavelet(timeS, dtS, f0)
q = judiVector(srcGeometry, wavelet)

# set up simulation operators
opt = Options(return_array=true)
Fs = [judiModeling(models[i], srcGeometry, recGeometry; options=opt) for i = 1:nv] # acoustic wave equation solver

## wave physics
function F(v::Vector{Matrix{Float32}})
    m = [vec(1f3./v[i]).^2f0 for i = 1:nv]
    return [Fs[i](m[i], q) for i = 1:nv]
end

global d_obs = [Fs[i]*q for i = 1:nv]

# Main loop
niterations = 50
fhistory = zeros(niterations)
fnoerror = zeros(niterations)

## initial
K0 = mean(perm, dims=3)[:,:,1]

## load FNO
device = cpu
net_path_FNO = datadir("trained-net", "trained-FNO.jld2")
net_dict_FNO = JLD2.jldopen(net_path_FNO, "r")
NN = net_dict_FNO["NN_save"] |> device;
AN = net_dict_FNO["AN"] |> device;
grid_ = gen_grid(net_dict_FNO["n"], net_dict_FNO["d"], net_dict_FNO["nt"], net_dict_FNO["dt"]) |> device;
Flux.testmode!(NN, true);

function SFNO(x)
    return relu01(NN(perm_to_tensor(x, grid_, AN)))[:,:,:,1];
end

K0 = K0 |> device;
JLD2.@load datadir("examples", "K.jld2") K
@time y_init = SFNO(K0) |> cpu;
@time y_true = SFNO(K |> device);

state_true = Saturations(state) |> device
println("FNO prediction error on true = ", norm(vec(y_true)-state_true)/norm(state_true))

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

K0 = mean(perm, dims=3)[:,:,1] |> device;
K0 += noiseLev * randn(Float32, size(K0)) * 120f0 |> device
z = vec(G1.inverse(reshape(K0,n[1],n[end],1,1)));

ls = BackTracking(c_1=1f-4,iterations=10,maxstep=Inf32,order=3,ρ_hi=5f-1,ρ_lo=1f-1)
### mask direct arrival
d_obs = [reshape(d_obs[i], ntR, nrec, 1, nsrc) for i = 1:nv]
wb = [find_water_bottom(d_obs[i][:,:,1,j]') for i = 1:nv, j = 1:nsrc]
for i = 1:nv
    for j = 1:nsrc
        wb[i,j] .+= 50
    end
end
data_mask = 0f0 * d_obs
for i = 1:nv
    for j = 1:nsrc
        for k = 1:nrec
            data_mask[i][wb[i,j][k]:end,k,1,j] .= 1
        end
    end
end

## add noise
noise_ = deepcopy(d_obs)
for i = 1:nv
    noise_[i] = randn(eltype(d_obs[i]), size(d_obs[i]))
end
snr = 10f0
noise_ = noise_/norm(noise_) *  norm(d_obs) * 10f0^(-snr/20f0)
d_obs = d_obs + noise_

function obj_ad(z)
    global K0 = G1(z)[:,:,1,1]
    global c = O(SFNO(K0)|> cpu); v = R(c); v_up = u(v); dpred = F(v_up);
    dpred_mask = [data_mask[i] .* dpred[i] for i = 1:nv]
    dobs_mask = [data_mask[i] .* d_obs[i] for i = 1:nv]
    fval = .5f0 * norm(dpred_mask-dobs_mask)^2f0
    @show fval
    return fval
end

function obj(z)
    global K0 = G1(z)[:,:,1,1]
    global c = O(SFNO(K0)|> cpu); v = R(c); v_up = u(v); dpred = F(v_up);
    dpred_ = [reshape(dpred[i], ntR, nrec, 1, nsrc) for i = 1:nv]
    dpred_mask = [data_mask[i] .* dpred_[i] for i = 1:nv]
    dobs_mask = [data_mask[i] .* d_obs[i] for i = 1:nv]
    fval = .5f0 * norm(dpred_mask-dobs_mask)^2f0
    @show fval
    return fval
end

K_init = deepcopy(K0|>cpu)
τ = 0.6f0
β = 1.2f0
τinit = deepcopy(τ)
τend = 1.2f0
z = prjz(z; α=0f0)
K0 = box_K(G1(z)[:,:,1,1])
step = 5f-1
for j=1:niterations

    Base.flush(Base.stdout)   
    co2true = Float32.(Saturations(S(T(Float64.(log.(K0*md)|>cpu)), f))|>device)
    fnoerror[j] = norm(vec(SFNO(K0))-co2true)/norm(co2true)
    @time fval, gs = Flux.withgradient(() -> obj_ad(z), Flux.params(z))
    g = gs[z]
    fhistory[j] = fval
    p = -g/norm(g, Inf)
    
    println("Inversion iteration no: ",j,"; function value: ", fhistory[j])

    # linesearch
    function f_(α_)
        misfit = obj(Float32.(prjz(z .+ α_ * p; α=τ)))
        @show α_, misfit
        return misfit
    end

    global step, fval = ls(f_, 1.2f0 * step, fval, dot(g, p))

    # Update model and bound projection
    global z = Float32.(prjz(z .+ step .* p; α=τ))
    global K0 = box_K(G1(z)[:,:,1,1])
    global logK0 = box_logK(log.(box_K(K0 .* Float32(md))))

    ### plotting
    y_predict = box_co2(O(SFNO(K0)|> cpu));

    K0_save = K0 |> cpu
    ### save intermediate results
    save_dict = @strdict β τinit τend j snr K0_save step niterations nv nsrc nrec survey_indices fhistory fnoerror
    @tagsave(
        joinpath(datadir(sim_name, exp_name), savename(save_dict, "jld2"; digits=6)),
        save_dict;
        safe=true
    )

    ## save figure
    fig_name = @strdict β τinit τend j snr niterations nv nsrc nrec survey_indices

    ## compute true and plot
    SNR = -2f1 * log10(norm(K-(K0|>cpu))/norm(K))
    fig = figure(figsize=(20,12));
    subplot(2,2,1);
    imshow((K0 |> cpu)',vmin=20,vmax=120);title("inversion by NN, $(j) iter");colorbar();
    subplot(2,2,2);
    imshow(K',vmin=20,vmax=120);title("GT permeability");colorbar();
    subplot(2,2,3);
    imshow(K_init',vmin=20,vmax=120);title("initial permeability");colorbar();
    subplot(2,2,4);
    imshow(abs.(K'-(K0|>cpu)'),vmin=20,vmax=120);title("error, SNR=$SNR");colorbar();
    suptitle("End-to-end Inversion at iter $j, seismic data snr=$snr")
    tight_layout()
    safesave(joinpath(plotsdir(sim_name, exp_name), savename(fig_name; digits=6)*"_K.png"), fig);
    close(fig)

    ## loss
    fig = figure(figsize=(20,12));
    subplot(1,2,1)
    plot(fhistory[1:j]);title("loss=$(fhistory[j])");
    subplot(1,2,2)
    plot(fnoerror[1:j]);title("fno prediction error=$(fnoerror[j])");
    suptitle("End-to-end Inversion at iter $j, seismic data snr=$snr")
    tight_layout()
    safesave(joinpath(plotsdir(sim_name, exp_name), savename(fig_name; digits=6)*"_loss.png"), fig);
    close(fig)

    ## data fitting
    fig = figure(figsize=(20,12));
    for i = 1:4
        subplot(4,4,i);
        imshow(y_init[:,:,i]', vmin=0, vmax=1);
        title("initial prediction at snapshot $(survey_indices[i])")
        subplot(4,4,i+4);
        imshow(sw_true[i,:,:]', vmin=0, vmax=1);
        title("true at snapshot $(survey_indices[i])")
        subplot(4,4,i+8);
        imshow(y_predict[i]', vmin=0, vmax=1);
        title("predict at snapshot $(survey_indices[i])")
        subplot(4,4,i+12);
        imshow(5*abs.(sw_true[i,:,:]'-y_predict[i]'), vmin=0, vmax=1);
        title("5X diff at snapshot $(survey_indices[i])")
    end
    suptitle("End-to-end Inversion at iter $j, seismic data snr=$snr")
    tight_layout()
    safesave(joinpath(plotsdir(sim_name, exp_name), savename(fig_name; digits=6)*"_saturation.png"), fig);
    close(fig)
    if j%(div(niterations, 3*Int(floor(log(τend/τinit)/log(β))+1))) == 0
        global τ = τ * β
        global τ = min(τ, τend)
    end
end
