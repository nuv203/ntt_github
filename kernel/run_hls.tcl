# run_hls.tcl — Vitis HLS (vitis_hls -f run_hls.tcl)
# Run from the kernel/ directory.
# Generates: ntt_project/solution1/syn/report/csynth.xml

open_project -reset ntt_project
set_top ntt_kernel
add_files ntt.cpp -cflags "-std=c++14"
add_files -tb ntt_tb.cpp -cflags "-std=c++14"

open_solution -reset solution1
set_part {xck26-sfvc784-2LV-c}
create_clock -period 10

if {[catch {csim_design} res]} {
    puts "ERROR: C-Simulation failed."
    puts $res
    exit 1
}
puts "C-Simulation passed."

if {[catch {csynth_design} res]} {
    puts "ERROR: C-Synthesis failed."
    puts $res
    exit 1
}
puts "C-Synthesis complete. Report: ntt_project/solution1/syn/report/csynth.xml"
exit 0
