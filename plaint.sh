#!/bin/sh
julia -p 2 --math-mode=ieee src/plaint.jl "$@"
