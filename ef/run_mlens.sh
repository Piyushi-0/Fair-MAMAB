n=6039
k=18
for T in 100000
    do
        for alpha in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.67 0.7 0.8 0.9 1
        do
            for SEED in $(seq 1 100)
            do
                python ef_mlens.py --alpha $alpha --T $T --SEED $SEED --save_as logs_ef_mlens --n $n --k $k --c_idx 0
            done
        done
    done