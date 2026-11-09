cmake_minimum_required(VERSION 3.16)
project(asr_host CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Same ORT location convention as tts-engine/host: a vendored x86 ORT 1.22.1.
# For the aarch64/musl build we cross-compile by hand against the Alpine
# rootfs (see ASR.md), not through this file.
set(ORT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../tts-engine/host/ort CACHE PATH "onnxruntime root")
add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
  IMPORTED_LOCATION ${ORT_DIR}/lib/libonnxruntime.so
  INTERFACE_INCLUDE_DIRECTORIES ${ORT_DIR}/include)

add_executable(asr_host src/asr_host.cpp)
target_link_libraries(asr_host PRIVATE onnxruntime pthread)
target_compile_options(asr_host PRIVATE -O2 -Wall)
set_target_properties(asr_host PROPERTIES
  BUILD_RPATH ${ORT_DIR}/lib INSTALL_RPATH ${ORT_DIR}/lib)
