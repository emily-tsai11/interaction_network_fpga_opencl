import matplotlib.pyplot as plt
import numpy as np
import time

def read_input():
	# get file contents
	with open('out/fpga_run.out', 'r') as f:
		file_contents = f.readlines()

	front_buffer_lines = 9
	middle_buffer_lines = 3
	num_data_files = 5
	num_time_lines = 56

	files = []
	nn_cpu_operation = ['define ints', 'create intermediate buffers', 'define dtypes/cl_halfs', 'flatten2dvec2arrays', 'create intermediate buffers']
	nn_cpu_time = [0.0, 0.0, 0.0, 0.0, 0.0]
	cpu_overhead_time = []
	kernel_name = ['transpose', 'buf_fastMatMul', 'buf_fastMatMul', 'interaction_cat', 'transpose', 'linear', 'linear', 'linear', 'linear', 'buf_fastMatMul', 'transpose', 'aggregate_cat', 'transpose', 'linear', 'linear', 'linear', 'transpose', 'buf_fastMatMul', 'buf_fastMatMul', 'interaction_cat', 'transpose', 'linear', 'linear', 'linear', 'linear']
	kernel_execution_time = []
	for i in range((num_time_lines - 2 * middle_buffer_lines) / 2):
		cpu_overhead_time.append(0.0)
		kernel_execution_time.append(0.0)
	avg_var_per_event_time = []
	factor = 1.0 # keep in microseconds

	# parse file
	for i in range(num_data_files):
		for j in range(num_time_lines):
			index = i * (num_time_lines + middle_buffer_lines) + j + front_buffer_lines
			temp_contents = file_contents[index].strip().split()

			if temp_contents[0] == 'Time':
				if len(temp_contents) == 7:
					temp_operation = temp_contents[4][1:-2]
				elif len(temp_contents) == 8:
					temp_operation = temp_contents[4][1:] + ' ' + temp_contents[5][:-2]
				elif len(temp_contents) == 9:
					temp_operation = temp_contents[4][1:] + ' ' + temp_contents[5] + ' ' + temp_contents[6][:-2]
				temp_operation_runtime = temp_contents[-2]
				# print(temp_operation + ' runtime: ' + temp_operation_runtime + ' microseconds')
			elif len(temp_contents) == 1:
				files.append(file_contents[index].strip())
				# print(file_contents[index].strip())
			elif i % 2 == 0:
				index_j = (j - 2 * middle_buffer_lines) / 2
				if index % 2 == 0:
					# print('CPU overhead: ' + temp_contents[5] + ' microseconds')
					cpu_overhead_time[index_j] += float(temp_contents[5]) / factor / num_data_files
				elif index % 2 == 1:
					# temp_kernel_name = temp_contents[0][:-1]
					# temp_kernel_runtime = temp_contents[-2]
					# print(temp_kernel_name + ' kernel runtime: ' + temp_kernel_runtime + ' microseconds')
					kernel_execution_time[index_j] += float(temp_contents[-2]) / factor / num_data_files
			elif i % 2 == 1:
				if index % 2 == 1:
					# print('CPU overhead: ' + temp_contents[5] + ' microseconds')
					cpu_overhead_time[index_j] += float(temp_contents[5]) / factor / num_data_files
				elif index % 2 == 0:
					# temp_kernel_name = temp_contents[0][:-1]
					# temp_kernel_runtime = temp_contents[-2]
					# print(temp_kernel_name + ' kernel runtime: ' + temp_kernel_runtime + ' microseconds')
					kernel_execution_time[index_j] += float(temp_contents[-2]) / factor / num_data_files

		# print('-------------------------------------------------')

	avg_var_per_event_time.append(float(file_contents[-4].strip().split()[-2]) / factor)
	avg_var_per_event_time.append(float(file_contents[-3].strip().split()[-2]) / factor)
	avg_var_per_event_time.append(float(file_contents[-2].strip().split()[-2]) / factor)
	avg_var_per_event_time.append(float(file_contents[-1].strip().split()[-2]) / factor)
	# print('Total per event, avg: ' + file_contents[-4].strip().split()[-2] + ' microseconds')
	# print('Total per event, var: ' + file_contents[-3].strip().split()[-2] + ' microseconds')
	# print('FPGA per event, avg: ' + file_contents[-2].strip().split()[-2] + ' microseconds')
	# print('FPGA per event, var: ' + file_contents[-1].strip().split()[-2] + ' microseconds')

	return files, nn_cpu_operation, nn_cpu_time, cpu_overhead_time, kernel_name, kernel_execution_time, avg_var_per_event_time

def get_forward_pass_times(kernel_execution_time, cpu_overhead_time):
	forward_pass_labels = ['$O^T$', 'S=$O^T$SR', 'R=$O^T$RR', 'I=cat(S,R,RI)', 'E=RM(I)', 'Er=(RR E)$^T$', 'agg=cat($O^T$,Er)', 'Pred=OM(agg)$^T$', 'S=Pred SR', 'R=Pred RR', 'I=cat(S,R,RI)', 'Pred=RM(I)']
	forward_pass = []
	forward_pass_cpu_overhead = []

	# O^T
	forward_pass.append(kernel_execution_time[0])
	forward_pass_cpu_overhead.append(cpu_overhead_time[0])

	# S=$O^T$SR
	forward_pass.append(kernel_execution_time[1])
	forward_pass_cpu_overhead.append(cpu_overhead_time[1])

	# R=$O^T$RR
	forward_pass.append(kernel_execution_time[2])
	forward_pass_cpu_overhead.append(cpu_overhead_time[2])

	# I=cat(S,R,RI)
	forward_pass.append(kernel_execution_time[3])
	forward_pass_cpu_overhead.append(cpu_overhead_time[3])

	# E=RM(I)
	forward_pass.append(kernel_execution_time[4] + kernel_execution_time[5] + kernel_execution_time[6] + kernel_execution_time[7] + kernel_execution_time[8])
	forward_pass_cpu_overhead.append(cpu_overhead_time[4] + cpu_overhead_time[5] + cpu_overhead_time[6] + cpu_overhead_time[7] + cpu_overhead_time[8])

	# Er=(RR E)$^T$
	forward_pass.append(kernel_execution_time[9] + kernel_execution_time[10])
	forward_pass_cpu_overhead.append(cpu_overhead_time[9] + cpu_overhead_time[10])

	# agg=cat($O^T$,Er)
	forward_pass.append(kernel_execution_time[11])
	forward_pass_cpu_overhead.append(cpu_overhead_time[11])

	# Pred=OM(agg)$^T$
	forward_pass.append(kernel_execution_time[12] + kernel_execution_time[13] + kernel_execution_time[14] + kernel_execution_time[15] + kernel_execution_time[16])
	forward_pass_cpu_overhead.append(cpu_overhead_time[12] + cpu_overhead_time[13] + cpu_overhead_time[14] + cpu_overhead_time[15] + cpu_overhead_time[16])

	# S=Pred SR
	forward_pass.append(kernel_execution_time[17])
	forward_pass_cpu_overhead.append(cpu_overhead_time[17])

	# R=Pred RR
	forward_pass.append(kernel_execution_time[18])
	forward_pass_cpu_overhead.append(cpu_overhead_time[18])

	# I=cat(S,R,RI)
	forward_pass.append(kernel_execution_time[19])
	forward_pass_cpu_overhead.append(cpu_overhead_time[19])

	# Pred=RM(I)
	forward_pass.append(kernel_execution_time[20] + kernel_execution_time[21] + kernel_execution_time[22] + kernel_execution_time[23] + kernel_execution_time[24])
	forward_pass_cpu_overhead.append(cpu_overhead_time[20] + cpu_overhead_time[21] + cpu_overhead_time[22] + cpu_overhead_time[23] + cpu_overhead_time[24])

	return forward_pass_labels, forward_pass, forward_pass_cpu_overhead

if __name__ == '__main__':
	start = time.time()

	files, nn_cpu_operation, nn_cpu_time, cpu_overhead_time, kernel_name, kernel_execution_time, avg_var_per_event_time = read_input()

	cpu_overhead_and_kernel_execution_time = []
	for i in range(len(cpu_overhead_time)):
		cpu_overhead_and_kernel_execution_time.append(cpu_overhead_time[i] + kernel_execution_time[i])

	plt.figure(0)
	plt.gcf().subplots_adjust(bottom = 0.26)
	plt.plot(np.linspace(0, len(kernel_execution_time), len(kernel_execution_time)), kernel_execution_time, '.', markersize = 10.0, label = 'Runtime')
	plt.legend(loc = 'best')
	plt.title('Runtime of Kernel Operations on FPGA')
	plt.xlabel('Kernel Operation')
	plt.ylabel('Runtime (microseconds)')
	plt.xticks(np.linspace(0, len(kernel_execution_time), len(kernel_execution_time)), kernel_name)
	plt.xticks(rotation = 70)
	plt.xlim([-1, len(kernel_execution_time) + 1])
	plt.savefig('out/benchmark_all_fpga.png')

	plt.plot(np.linspace(0, len(cpu_overhead_and_kernel_execution_time), len(cpu_overhead_and_kernel_execution_time)), cpu_overhead_and_kernel_execution_time, '.', markersize = 10.0, label = 'CPU Overhead + Runtime')
	plt.title('Runtime of Kernel Operations on FPGA with CPU Overhead')
	plt.legend(loc = 'best')
	plt.savefig('out/benchmark_all_cpu_fpga.png')

	forward_pass_labels, forward_pass, forward_pass_cpu_overhead = get_forward_pass_times(kernel_execution_time, cpu_overhead_time)
	forward_pass_cpu_and_kernel_time = []
	for i in range(len(forward_pass)):
		forward_pass_cpu_and_kernel_time.append(forward_pass[i] + forward_pass_cpu_overhead[i])

	plt.figure(1)
	plt.gcf().subplots_adjust(bottom = 0.28)
	plt.plot(np.linspace(0, len(forward_pass), len(forward_pass)), forward_pass, '.', markersize = 10.0, label = 'Runtime')
	plt.legend(loc = 'best')
	plt.title('Runtime of IN Forward Pass on FPGA')
	plt.xlabel('Operation')
	plt.ylabel('Runtime (microseconds)')
	plt.xticks(np.linspace(0, len(forward_pass), len(forward_pass)), forward_pass_labels)
	plt.xticks(rotation = 70)
	plt.xlim([-1, len(forward_pass) + 1])
	plt.savefig('out/benchmark_forward_pass_fpga.png')

	plt.plot(np.linspace(0, len(forward_pass_cpu_and_kernel_time), len(forward_pass_cpu_and_kernel_time)), forward_pass_cpu_and_kernel_time, '.', markersize = 10.0, label = 'CPU Overhead + Runtime')
	plt.title('Runtime of IN Forward Pass on FPGA with CPU Overhead')
	plt.legend(loc = 'best')
	plt.savefig('out/benchmark_forward_pass_cpu_fpga.png')

	print('Runtime of benchmark_plots.py: ' + str(time.time() - start) + ' seconds')
