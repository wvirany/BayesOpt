TARGETS=("PARP1" "F2" "ESR2" "PGR")
N_INITS=(100 1000)
RADII=(2 4)
SPARSE_OPTIONS=("true" "false")
POOL=1000000
BUDGET=1000

for TARGET in "${TARGETS[@]}"; do
    for N_INIT in "${N_INITS[@]}"; do
        for RADIUS in "${RADII[@]}"; do
        echo "Generating plot for: ${TARGET}, pool=${POOL}, n_init=${N_INIT}, radius=${RADIUS}"

            python3 plot-bo-results.py --target ${TARGET} \
            --pool ${POOL} \
            --n_init ${N_INIT} \
            --budget ${BUDGET} \
            --radius ${RADIUS}
        
        done
    done
done