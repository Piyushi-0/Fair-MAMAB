n=4
k=3
for T in 100000
    do
    for A_choice in 0 6 19 21 23
        do
        for c in 0.3 0.1  #0.5 0.3 0.1
            do
                for SEED in $(seq 1 100)
                do
                    echo $A_choice $c
                    python dual_lambda1.py --T $T --n $n --k $k --SEED $SEED --A_choice $A_choice --save_as logs_dual_lambda1 --c $c
                done
            done
        done
    done