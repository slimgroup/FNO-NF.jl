## Author: Ziyi Yin, ziyi.yin@gatech.edu
## Date: Sep 17, 2023

## Permeability inversion
## Observed data: seismic + well
## Methods: unconstrained optimization with numerical simulator

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
using Flux
using Random
using LineSearches
using Statistics
using FNO4CO2
using JUDI
using InvertibleNetworks
Random.seed!(2023)

matplotlib.use("agg")
include(srcdir("utils.jl"))

sim_name = "combine-inversion"
exp_name = "jutul"

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


obj(logK) = .5 * norm(S(T(prj(logK)), f)[1:length(tstep)*prod(n)]-state[1:length(tstep)*prod(n)])^2f0

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
O(state::AbstractVector) = Float32.(permutedims(reshape(state[1:length(tstep)*prod(n)], n[1], n[end], length(tstep)), [3,1,2])[survey_indices,:,:])
sw_true = O(state)

# set up rock physics
vp = 3500 * ones(Float32,n[1],n[end])     # p-wave
phi = 0.25f0 * ones(Float32,n[1],n[end])  # porosity
rho = 2200 * ones(Float32,n[1],n[end])    # density
R(c::AbstractArray{Float32,3}) = Patchy(c,vp,rho,phi)[1]
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

ls = BackTracking(order=3, iterations=20)

# Main loop
niterations = 50
fhistory = zeros(niterations)

## initial
K0 = mean(perm, dims=3)[:,:,1] * md
logK0 = log.(K0)
logK_init = deepcopy(logK0)

@time state0 = S(T(logK0), f)

y_init = box_co2(O(S(T(logK_init), f)))

function obj_wave_ad(logK)
    c = box_co2(O(S(T(logK), f))); v = R(c); v_up = box_v(u(v)); dpred = F(v_up);
    dpred_mask = [data_mask[i] .* dpred[i] for i = 1:nv]
    dobs_mask = [data_mask[i] .* d_obs[i] for i = 1:nv]
    fval = .5f0 * norm(dpred_mask-dobs_mask)^2f0
    return fval
end

function obj_wave(logK)
    c = box_co2(O(S(T(logK), f))); v = R(c); v_up = box_v(u(v)); dpred = F(v_up);
    dpred_ = [reshape(dpred[i], ntR, nrec, 1, nsrc) for i = 1:nv]
    dpred_mask = [data_mask[i] .* dpred_[i] for i = 1:nv]
    dobs_mask = [data_mask[i] .* d_obs[i] for i = 1:nv]
    fval = .5f0 * norm(dpred_mask-dobs_mask)^2f0
    return fval
end

obs_loc = zeros(Float32, n[1], n[end], length(tstep))
subsamp = 3
obs_loc[Int.(round.(range(1, stop=n[1], length=subsamp))),:,survey_indices] .= 1f0
obs_loc = vec(obs_loc)
obj_well(logK) = .5 * norm(obs_loc .* (S(T(logK), f)[1:length(tstep)*prod(n)]-state[1:length(tstep)*prod(n)]))^2f0

λ = 1f1
obj(x) = obj_wave(x) + λ * obj_well(x)
obj_ad(x) = obj_wave_ad(x) + λ * obj_well(x)

global step = 1e-2
for j=1:niterations

    Base.flush(Base.stdout)   
    @time fval, gs = Flux.withgradient(() -> obj_ad(logK0), Flux.params(logK0))
    g = gs[logK0]
    fhistory[j] = fval
    p = -g
    
    println("Inversion iteration no: ",j,"; function value: ", fhistory[j])

    # linesearch
    function f_(α)
        misfit = obj(box_logK(logK0 .+ α .* p))
        @show α, misfit
        return misfit
    end

    global step, fval = ls(f_, 10f0 * step, fhistory[j], dot(g, p))

    # Update model and bound projection
    global logK0 = box_logK(logK0 .+ step .* p)

    ### plotting
    y_predict = box_co2(O(S(T(logK0), f)))

    ### save intermediate results
    save_dict = @strdict λ subsamp j snr logK0 step niterations nv nsrc nrec survey_indices fhistory
    @tagsave(
        joinpath(datadir(sim_name, exp_name), savename(save_dict, "jld2"; digits=6)),
        save_dict;
        safe=true
    )

    ## save figure
    fig_name = @strdict λ subsamp j snr niterations nv nsrc nrec survey_indices

    ## compute true and plot
    SNR = -2f1 * log10(norm(K-exp.(logK0))/norm(K))
    fig = figure(figsize=(20,12));
    subplot(2,2,1);
    imshow(exp.(logK0)'./md,vmin=20,vmax=120);title("inversion by NN, $(j) iter");colorbar();
    subplot(2,2,2);
    imshow(K'./md,vmin=20,vmax=120);title("GT permeability");colorbar();
    subplot(2,2,3);
    imshow(exp.(logK_init)'./md,vmin=20,vmax=120);title("initial permeability");colorbar();
    subplot(2,2,4);
    imshow(abs.(K'-exp.(logK0)')./md,vmin=20,vmax=120);title("error, SNR=$SNR");colorbar();
    suptitle("End-to-end Inversion at iter $j, seismic data snr=$snr")
    tight_layout()
    safesave(joinpath(plotsdir(sim_name, exp_name), savename(fig_name; digits=6)*"_K.png"), fig);
    close(fig)

    ## loss
    fig = figure(figsize=(20,12));
    plot(fhistory[1:j]);title("loss=$(fhistory[j])");
    suptitle("End-to-end Inversion at iter $j, seismic data snr=$snr")
    tight_layout()
    safesave(joinpath(plotsdir(sim_name, exp_name), savename(fig_name; digits=6)*"_loss.png"), fig);
    close(fig)

    ## data fitting
    fig = figure(figsize=(20,12));
    for i = 1:4
        subplot(4,4,i);
        imshow(y_init[i,:,:]', vmin=0, vmax=1);
        title("initial prediction at snapshot $(survey_indices[i])")
        subplot(4,4,i+4);
        imshow(sw_true[i,:,:]', vmin=0, vmax=1);
        title("true at snapshot $(survey_indices[i])")
        subplot(4,4,i+8);
        imshow(y_predict[i,:,:]', vmin=0, vmax=1);
        title("predict at snapshot $(survey_indices[i])")
        subplot(4,4,i+12);
        imshow(5*abs.(sw_true[i,:,:]'-y_predict[i,:,:]'), vmin=0, vmax=1);
        title("5X diff at snapshot $(survey_indices[i])")
    end
    suptitle("End-to-end Inversion at iter $j, seismic data snr=$snr")
    tight_layout()
    safesave(joinpath(plotsdir(sim_name, exp_name), savename(fig_name; digits=6)*"_saturation.png"), fig);
    close(fig)

end