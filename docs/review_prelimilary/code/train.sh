config=sia_pvt_daf
for f in {0..4}; do
	sed -i -E -e "s/((fold|version|load_from).*)[0-4]([^0-9]*)$/\1$f\3/g" ./configs/${config}.yaml
	python ./Solver.py --config ./configs/${config}.yaml --gpus 2,3
done