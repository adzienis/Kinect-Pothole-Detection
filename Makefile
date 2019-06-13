centroids:
	mpiexec  --use-hwthread-cpus -np 1 python3 ./driver.py : -np 3 ./kmeans

no_centroids:
	python3 driver_no_centroids.py
