n=6039
k=18
T=100000
for SEED in $(seq 17 100)
do
    python dual_lambda1_mlens.py --T $T --SEED $SEED --save_as logs_dual_lambda1_mlens --c 0 --n $n --k $k
done
