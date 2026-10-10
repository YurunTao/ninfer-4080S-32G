# Compile policy is owned here; each semantic family explicitly contributes sources.
add_library(ninfer_ops STATIC)
ninfer_internal_includes(ninfer_ops)
ninfer_cuda_archive(ninfer_ops)

if(NINFER_ARCH_SM89)
  # NVFP4 A4 execution needs Blackwell block-scale MMA and Hopper-or-newer TMA,
  # neither of which exists on sm_89. The families still list their sm_120a
  # translation units on ninfer_nvfp4_non_rdc, so that target degrades to an
  # interface library here and those units are never compiled. The branch's
  # rejection stubs keep the public entry points defined and linking.
  add_library(ninfer_nvfp4_non_rdc INTERFACE)
  target_sources(ninfer_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/nvfp4_sm89_stubs.cpp")
else()
  # Warp-specialized SM120 NVFP4 kernels use setmaxnreg to transfer registers from
  # producer to consumer warpgroups. RDC makes ptxas discard that contract. These
  # self-contained kernels therefore use a non-RDC archive and host launchers.
  add_library(ninfer_nvfp4_non_rdc STATIC)
  ninfer_internal_includes(ninfer_nvfp4_non_rdc)
  ninfer_cuda_non_rdc_archive(ninfer_nvfp4_non_rdc)
  target_link_libraries(ninfer_nvfp4_non_rdc PRIVATE ninfer_core CUDA::cudart CUDA::cuda_driver)
endif()

target_link_libraries(ninfer_ops PUBLIC ninfer_core PRIVATE ninfer_nvfp4_non_rdc)

include(basic_sources.cmake)
include("${CMAKE_CURRENT_LIST_DIR}/attn_input_proj/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/candidate_selector/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/context_kv_materialize/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/dynamic_grouped_conv/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/gdn_gating_proj/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/gdn_input_proj/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/kv_cache/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_add/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_attention/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_pair/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_swiglu/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_topk/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/rmsnorm_rope/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/softmax_attention/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/sparse_moe/sources.cmake")
