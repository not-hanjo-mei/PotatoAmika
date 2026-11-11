cmake_minimum_required(VERSION 3.20)
project(potato_tts_host CXX)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)  # stage 3: kernels need -O3; -O0 cost 5-10x
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(ORT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ort)
add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
  IMPORTED_LOCATION ${ORT_DIR}/lib/libonnxruntime.so
  INTERFACE_INCLUDE_DIRECTORIES ${ORT_DIR}/include)

# POTATO_FREE_ONLY: build tts_host with the ORT engine path compiled out
# (main.cpp guards on the macro of the same name; packed.h/vits_packed.h
# are bypassed for packed_core.h) and no onnxruntime linkage. Default
# OFF — the shipping dual-engine build is unchanged. See ANDROID-SPIKE.md.
option(POTATO_FREE_ONLY "tts_host: free engine only, no ONNX Runtime" OFF)

add_executable(tts_host src/main.cpp src/g2p.cpp src/vits_kernels.cpp src/free_engine.cpp)
if(POTATO_FREE_ONLY)
  target_compile_definitions(tts_host PRIVATE POTATO_FREE_ONLY=1)
  if(NOT ANDROID)
    # bionic has pthread inside libc; there is no separate -lpthread.
    target_link_libraries(tts_host PRIVATE pthread)
  endif()
else()
  target_link_libraries(tts_host PRIVATE onnxruntime)
  set_target_properties(tts_host PROPERTIES
    BUILD_RPATH ${ORT_DIR}/lib
    INSTALL_RPATH ${ORT_DIR}/lib)
endif()

add_executable(tf_check src/tf_check.cpp)
target_link_libraries(tf_check PRIVATE onnxruntime)

add_executable(vits_check src/vits_check.cpp src/vits_kernels.cpp)
target_link_libraries(vits_check PRIVATE onnxruntime)

add_executable(kbench src/kbench.cpp)
target_link_libraries(kbench PRIVATE onnxruntime pthread)

# ORT-free engine (ORT-FREE.md): GPT-side replacement, phase 1.
# Deliberately NOT linked against onnxruntime -- the point of the
# exercise. Parallel existence only; shipping defaults unchanged.
# Phase 2: vits_kernels.cpp joins for the packed conv kernels; its
# declarations moved to the ORT-free vits_kernels.h, so still no ORT
# headers or library in these binaries (verify with ldd).
add_executable(tf_check_free src/tf_check_free.cpp src/free_engine.cpp src/vits_kernels.cpp)
target_link_libraries(tf_check_free PRIVATE pthread)
target_include_directories(tf_check_free PRIVATE src)

# ORT-free vits validation (phase 2): vits_check's protocol on the free
# engine. Also deliberately ORT-free; cross-checks against the ORT path
# through vits_check --dump files.
add_executable(vits_check_free src/vits_check_free.cpp src/free_engine.cpp src/vits_kernels.cpp)
target_link_libraries(vits_check_free PRIVATE pthread)
target_include_directories(vits_check_free PRIVATE src)

add_executable(test_softclip src/test_softclip.cpp)
set_target_properties(tf_check PROPERTIES
  BUILD_RPATH ${ORT_DIR}/lib
  INSTALL_RPATH ${ORT_DIR}/lib)
