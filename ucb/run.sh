for T in 100000
    do
    for A_choice in 6 0 19 21 23
        do
        for c in 0.5 0.3 0.1
            do
                for SEED in $(seq 1 100)
                do
                    echo $A_choice $c
                    python ucb.py --T $T --SEED $SEED --A_choice $A_choice --save_as logs_ucb --c $c
                done
            done
        done
    done

n=4
k=3
for T in 100000
    do
    for A_choice in 0 1 2 3 4
        do
        for c in 0.5 0.3 0.1
            do
                for SEED in $(seq 1 100)
                do
                    echo $A_choice $c
                    python ucb.py --n $n --k $k --T $T --SEED $SEED --A_choice $A_choice --save_as logs_ucb --c $c
                done
            done
        done
    done