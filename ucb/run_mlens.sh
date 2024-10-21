n=6039
k=18
T=100000
for SEED in $(seq 1 100)
do
    python ucb_mlens.py --T $T --SEED $SEED --save_as logs_ucb_mlens --c 0 --n $n --k $k
done
