from multiprocessing import Process
import os

def nn():
	os.system('bash emulator_run.sh')

if __name__ == '__main__':
	os.system('rm out/benchmark_wrapper.out')
	p = Process(target = nn)
	p.start()
	while p.is_alive():
		os.system('(top -b -n 1 -u et7417 | grep gnn_fpga) >> out/benchmark_wrapper.out')
	p.join()
