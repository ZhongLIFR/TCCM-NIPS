for i in 0 1 2 3 4
do
  python RobustnessStudy.py --dimensions 2 --random_seed $i &
done
wait
for i in 0 1 2 3 4
do
  python RobustnessStudy.py --dimensions 10 --random_seed $i &
done
wait
for i in 0 1 2 3 4
do
  python RobustnessStudy.py --dimensions 20 --random_seed $i &
done
wait
for i in 0 1 2 3 4
do
  python RobustnessStudy.py --dimensions 50 --random_seed $i &
done
wait
for i in 0 1 2 3 4
do
  python RobustnessStudy.py --dimensions 100 --random_seed $i &
done