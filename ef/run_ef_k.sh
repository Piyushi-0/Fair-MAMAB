n=4
k=3
for T in 100000
    do
    for A_choice in 0
        do
        for c in 0.3
            do
                for alpha in 0.67 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1
                do
                    for SEED in $(seq 1 100)
                    do
                        python ef_k.py --alpha $alpha --T $T --SEED $SEED --A_choice $A_choice --save_as logs_ef --c $c --n $n --k $k
                    done
                done
            done
        done
    done

# n=4
# k=3
# for T in 100000
#     do
#     for A_choice in 0 1 2 3 4
#         do
#         for c in 0.33 0.1
#             do
#                 for alpha in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.67 0.7 0.8 0.9 1
#                 do
#                     for SEED in $(seq 1 100)
#                     do
#                         python ef_k.py --alpha $alpha --T $T --SEED $SEED --A_choice $A_choice --save_as logs_ef --c $c --n $n --k $k
#                     done
#                 done
#             done
#         done
#     done

# c_idx=1
# for T in 100000
#     do
#     for A_choice in 0 1 2 3 4
#         do
#             for alpha in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.67 0.7 0.8 0.9 1
#             do
#                 for SEED in $(seq 1 100)
#                 do
#                     python ef_k.py --alpha $alpha --T $T --SEED $SEED --A_choice $A_choice --save_as logs_ef_${c_idx} --n $n --k $k --c_idx $c_idx
#                 done
#             done
#         done
#     done