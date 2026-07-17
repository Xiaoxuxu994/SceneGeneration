file(REMOVE_RECURSE
  "lib/libascendc_kernels_npu_pcd_dist.pdb"
  "lib/libascendc_kernels_npu_pcd_dist.so"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/ascendc_kernels_npu_pcd_dist.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
