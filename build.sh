export OMP_PROC_BIND=spread
export OMP_PLACES=threads
cmake -S ./highOrderTFEM -B ./bin
cd bin 
make