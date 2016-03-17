# SDAccel command script.

# Define a solution name.
create_solution -name host -dir FPGA -force

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

add_files "rbf.hpp"
set_property file_type "c header files" [get_files "rbf.hpp"]

add_files "layer.hpp"
set_property file_type "c header files" [get_files "layer.hpp"]

add_files "util.hpp"
set_property file_type "c header files" [get_files "util.hpp"]

add_files "test.hpp"
set_property file_type "c header files" [get_files "test.hpp"]


build_system

package_system
