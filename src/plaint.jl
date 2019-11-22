#!/usr/bin/env julia
#
# usage:
#  script......... plaint.jl /path/to/system.yml
#  interactively.. julia -qi plaint.jl

using YAML              #read data file
using DelimitedFiles    #output results
using LinearAlgebra

"""
    ksnr(e, M, eps, max_iterations)

Solve Kepler's equation with NR method.
"""
function ksnr(e::Float64, M::Float64,
              eps::Float64 = 1e-16, nm::Int64 = 16)
    x = M + e * sin(M)
    d = (x - e * sin(x) - M) / (1 - e * cos(x))
    while abs(d) > eps && nm > 0
        x -= d
        d = (x - e * sin(x) - M) / (1 - e * cos(x))
        nm -= 1
    end
    return x
end

"""
    euler_rotd(Ω, i, ω)

Return Eulerian rotation.
"""
function euler_rotd(Ω::Float64, i::Float64, ω::Float64)
    return [ cosd(Ω) -sind(Ω) 0; sind(Ω) cosd(Ω) 0; 0 0 1 ] *
           [ 1 0 0; 0 cosd(i) -sind(i); 0 sind(i) cosd(i) ] *
           [ cosd(ω) -sind(ω) 0; sind(ω) cosd(ω) 0; 0 0 1 ]
end

"""
    oe_to_xv!(mu, j, orbital_elements, positions, velocities)

Convert orbital elements to cartesian vectors.
"""
function oe_to_xv!(mu::Float64, j::Int64,
                   oe::Array{Float64,2},
                   vr::Array{Float64,2}, vv::Array{Float64,2})
    a, e, i, Ω, ω, M = oe[:, j]
    E = ksnr(e, M * π / 180)
    r = a * (1 - e * cos(E))
    pof = [cos(E) - e, sqrt(1 - e^2) * sin(E), 0] * a
    vof = [  - sin(E), sqrt(1 - e^2) * cos(E), 0] * (sqrt(mu * a) / r)
    rot = euler_rotd(Ω, i, ω)
    vr[:, j] = rot * pof
    vv[:, j] = rot * vof
end

"""
    xv_to_oe!(mu, j, positions, velocities, orbital_elements)

Convert cartesian vectors to orbital elements.
"""
function xv_to_oe!(mu::Float64, j::Int64,
                   rt::Array{Float64,2}, vt::Array{Float64,2},
                   oe::Array{Float64,2})
    tiny = 1e-10
    vr, vv = rt[:, j], vt[:, j]
    r = norm(vr)
    v = norm(vv)
    vh = cross(vr, vv)
    h = norm(vh)
    vn = [-vh[2], vh[1], 0]
    n = norm(vn)
    ve = cross(vv, vh) / mu - vr / r
    a = 1 / (2 / r - v^2 / mu)
    e = norm(ve)
    i = acos(vh[3] / h)
    if e > tiny
        if n > tiny
            Ω = acos(vn[1] / n)
            if vn[2] < 0
                Ω = 2π - Ω
            end
            ω = acos(dot(ve, vn) / (e * n))
            if ve[3] < 0
                ω = 2π - ω
            end
        else
            i = Ω = .0
            ω = atan2(ve[2], ve[3])
            if cross(ve, vr)[3] < 0
                ω = 2π - ω
            end
        end
        nu = acos(dot(ve, vr)/(e * r))
        if dot(vr, vv) < 0
            nu = 2π - nu
        end
    else
        #not implemented
    end
    E = 2 * atan(sqrt((1 - e) / (1 + e)) * tan(nu / 2))
    M = E - e * sin(E)
    if M < 0
        M += 2π
    end
    i, Ω, ω, M = [i, Ω, ω, M] * 180 / π
    oe[:, j] = [a, e, i, Ω % 360, ω % 360, M % 360]
end

"""
    grav_a!(no_of_bodies, masses, positions, acceleration)

Compute gravitational acceleration.
!!! note
    The `a` should be multiplied with the appropriate gravitational constant.
"""
function grav_a!(n::Int64, m::Array{Float64,1},
                 x::Array{Float64,2}, a::Array{Float64,2})
    fill!(a, 0)
    @inbounds for i = 1:(n - 1), j = (i + 1):n
        R = x[:, i] - x[:, j]
        Rd3 = R / norm(R)^3
        a[:, i] -= m[j] * Rd3
        a[:, j] += m[i] * Rd3
    end
end

"""
    mksnr(dt, α, n, σ0, r0, eps, max_iterations)

Solve modified Kepler's equation with NR method.
!!! note
    Argument `α` is the inverse of semimajor axis.
"""
function mksnr(dt::Float64, α::Float64, n::Float64, s::Float64, r0::Float64,
               eps::Float64 = 1e-16, nm::Int64 = 16)
    r0a, s0a, ndt = 1 - r0 * α, s * sqrt(α), n * dt
    x = π
    d = (x - r0a * sin(x) - s0a * (cos(x) - 1) - ndt) /
            (1 - r0a * cos(x) + s0a * sin(x))
    while abs(d) > eps && nm > 0
        x -= d
        d = (x - r0a * sin(x) - s0a * (cos(x) - 1) - ndt) /
                (1 - r0a * cos(x) + s0a * sin(x))
        nm -= 1
    end
    return x
end

"""
    gauss_step!(dt, mu, no_of_bodies, positions, velocities)

Keplerian drift using Gauss' f, g functions.
"""
function gauss_step!(dt::Float64, mu::Float64, np::Int64,
                     rt::Array{Float64,2}, vt::Array{Float64,2})
    @inbounds @simd for i = 1:np
        vr0, vv0 = rt[:, i], vt[:, i]
        r0, v0 = norm(vr0), norm(vv0)
        α = 2 / r0 - v0^2 / mu
        a, n = 1 / α, sqrt(mu * α^3)
        s = dot(vr0, vv0) / sqrt(mu)
        E = mksnr(dt, α, n, s, r0)
        cose, sine = cos(E), sin(E)
        r1 = a + (r0 - a) * cose + sqrt(a) * s * sine
        a1ce = a * (cose - 1)
        ivr0, ivr1 = 1 / r0, 1 / r1
        ft = 1 + a1ce * ivr0
        gt = dt + (sine - E) / n
        fd = - a^2 * n * sine * ivr1 * ivr0
        #gd = 1 + a1ce * ivr1
        gd = (1 + gt * fd) / ft
        rt[:, i] = ft * vr0 + gt * vv0
        vt[:, i] = fd * vr0 + gd * vv0
    end
end

"""
    eam_dh(no_of_bodies, masses, positions, velocities)

Compute energy and angular momentum, with democratic heliocentric vectors.
"""
function eam_dh(np::Int64, m::Array{Float64,1},
                x::Array{Float64,2}, v::Array{Float64,2})
    et, am = norm(sum(m' .* v, dims = 2))^2 / 2, zeros(3)
    @inbounds @simd for i = 1:np
        et += m[i] * norm(v[:, i])^2 / 2
        et -= m[i] / norm(x[:, i])
        am += m[i] * cross(x[:, i], v[:, i])
    end
    @inbounds for i = 1:(np - 1), j = (i + 1):np
        et -= m[i] * m[j] / norm(x[:, i] - x[:, j])
    end
    return et, norm(am)
end

"""
    plaint_dhi(iostream, total_iterations, dt, step_iterations,
               mu, masses, orbital_elements)

Perform planetary integration using democratic heliocentric integrator.
"""
function plaint_dhi(of::Array{IOStream, 1},
                    kmax::Float64, dt::Float64, kpri::Float64, mu::Float64,
                    mp::Array{Float64,1}, oe::Array{Float64,2})
    mt = 1 / (sum(mp) + 1)
    np = size(mp, 1)
    an = Array{Float64}(undef, 3, np)
    x = similar(an)
    v = similar(an)
    @inbounds @simd for j = 1:np
        oe_to_xv!(mu, j, oe, x, v)
    end
    v .-= mt * sum(mp' .* v, dims = 2)                  #to_dh
    el0 = eam_dh(np, mp, x, v)
    println("Initial energy:           ", el0[1], "\n",
            "Initial angular momentum: ", el0[2], "\n")
    d2 = dt / 2
    @inbounds for k = 1:kmax
        try
            x .+= d2 * sum(mp' .* v, dims = 2)
            gauss_step!(d2, mu, np, x, v)
            grav_a!(np, mp, x, an)
            v .+= dt * an
            gauss_step!(d2, mu, np, x, v)
            x .+= d2 * sum(mp' .* v, dims = 2)
            if k % kpri == 0
                eln = eam_dh(np, mp, x, v)
                va = v .+ mt * sum(mp' .* v, dims = 2)  #to_ac
                writedlm(of[end], [collect(eln .- el0); collect(@. 1 - eln / el0)]')
                @simd for j = 1:np
                    xv_to_oe!(mu, j, x, va, oe)
                    writedlm(of[j], oe[:, j]')
                end
            end
        catch
            println("Error: Exiting on ", k, "\n")
            break
        end
    end
    close.(of)
end

"""
    data_get(file)

Get planetary system's data and integration options.
"""
function data_get(file::String)
    data = YAML.load_file(file)
    tmax::Float64 = data["options"]["tmax"]
    step::Float64 = data["options"]["step"]
    ever::Float64 = data["options"]["ever"]
    unit = "solar"
    try
        unit = data["system"]["units"]
    catch
    end
    ms = data["system"]["star_mass"]
    datp = data["system"]["planets"]
    sort!(datp, by = x -> x["orbital_elements"][1])
    np = size(datp, 1)
    name = Array{String}(undef, np)
    mp = Array{Float64}(undef, np)
    oe = Array{Float64}(undef, 6, np)
    @inbounds @simd for i = 1:np
        name[i] = data["system"]["star_name"] * " " * datp[i]["id"]
        mp[i] = datp[i]["mass"]
        oe[:, i] = datp[i]["orbital_elements"]
    end
    return tmax, step, ever, ms, name, mp, oe, unit
end

"""
    data_prep(tmax, step, ever,
              star_mass, file_names, masses, orbital_elements, units)

Prepare data and options to feed to the integrator.
"""
function data_prep(tmax::Float64, step::Float64, ever::Float64,
                   ms::Float64, name::Array{String, 1},
                   mp::Array{Float64, 1}, oe::Array{Float64,2},
                   unit::String = "solar")
    if unit == "solar"
        mp = mp * 9.547919e-4 / ms
    else #natural
        mp = mp / ms
    end
    oe[1, :] /= oe[1, 1]
    tp = 2π
    dt = step * tp
    kmax = div(tmax * tp, dt)
    kpri = div(ever * tp, dt)
    of = map(f -> open("out/$f.txt", "w"), name)
    push!(of, open("out/eamerr.txt", "w"))
    println("Total simulation time is: ", tmax, " yr\n",
            "Time of every step to be: ", step, " yr\n",
            "Total number of steps is: ", kmax, " steps\n",
            "Output will be every:     ", ever, " yr\n",
            "or differently every:     ", kpri, " steps\n")
    return of, kmax, dt, kpri, 1.0, mp, oe
end

"""
    usage()

Return an exemplary session.
"""
function usage()
    println("## Example session:\n",
            "#set file containing data\n",
            "file = \"data/TRAPPIST-1.yml\";\n",
            "#get data from file\n",
            "tmax, step, ever, ms, name, mp, oe = data_get(file);\n",
            "#or enter manually\n",
            "dgr = (tmax, step, ever, ms, name, mp, oe);\n",
            "#prepare data\n",
            "of, kmax, dt, kpri, mu, mp, oe = data_prep(dgr...);\n",
            "#integrate, available: plaint_dhi, plaint_s6i\n",
            "plaint_dhi(of, kmax, dt, kpri, mu, mp, oe);\n",
            "#benchmark integration of a particular system\n",
            "@time plaint_dhi(data_prep(data_get(file)...)...);")
end

## Session
# Run as script or, interactively in a Julia session.

if ! isinteractive()
    @time plaint_dhi(data_prep(data_get(ARGS[1])...)...)
else
    usage()
end
