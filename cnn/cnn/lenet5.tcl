# SDAccel command script.

# Define a solution name.
create_solution -name lenet5 -dir FPGA -force

# Define the target platform of the application
add_device -vbnv xilinx:adm-pcie-7v3:1ddr:2.0

# Host source files.
add_files "main.cpp"

# Header files.
add_files "cnn.hpp"
set_property file_type "c header files" [get_files "cnn.hpp"]

add_files "convolution.hpp"
set_property file_type "c header files" [get_files "convolution.hpp"]

add_files "maxpool.hpp"
set_property file_type "c header files" [get_files "maxpool.hpp"]

add_files "fullconnect.hpp"
set_property file_type "c header files" [get_files "fullconnect.hpp"]

add_files "layer.hpp"
set_property file_type "c header files" [get_files "layer.hpp"]

add_files "util.hpp"
set_property file_type "c header files" [get_files "util.hpp"]

add_files "test.hpp"
set_property file_type "c header files" [get_files "test.hpp"]

# Create the kernel.
create_kernel conv1 -type clc
add_files -kernel [get_kernels conv1] "lenet5.cl"
create_kernel max2 -type clc
add_files -kernel [get_kernels max2] "lenet5.cl"
create_kernel conv3 -type clc
add_files -kernel [get_kernels conv3] "lenet5.cl"
create_kernel max4 -type clc
add_files -kernel [get_kernels max4] "lenet5.cl"
create_kernel conv5 -type clc
add_files -kernel [get_kernels conv5] "lenet5.cl"
create_kernel full6 -type clc
add_files -kernel [get_kernels full6] "lenet5.cl"

# Define binary containers.
create_opencl_binary alpha
set_property region "OCL_REGION_0" [get_opencl_binary alpha]
create_compute_unit -opencl_binary [get_opencl_binary alpha] -kernel [get_kernels conv1] -name CONV1
create_compute_unit -opencl_binary [get_opencl_binary alpha] -kernel [get_kernels max2] -name MAX2
create_compute_unit -opencl_binary [get_opencl_binary alpha] -kernel [get_kernels conv3] -name CONV3
create_compute_unit -opencl_binary [get_opencl_binary alpha] -kernel [get_kernels max4] -name MAX4
create_compute_unit -opencl_binary [get_opencl_binary alpha] -kernel [get_kernels conv5] -name CONV5
create_compute_unit -opencl_binary [get_opencl_binary alpha] -kernel [get_kernels full6] -name FULL6

# Compile the design for CPU based emulation.
compile_emulation -flow cpu -opencl_binary [get_opencl_binary alpha]

# Generate the system estimate report.
report_estimate

# Run the design in CPU emulation mode
run_emulation -flow cpu -args "../../../../../lenet5.xml result.xml alpha.xclbin"

build_system

package_system