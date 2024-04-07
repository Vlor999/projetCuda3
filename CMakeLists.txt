cmake_minimum_required(VERSION 3.16)
project(ass03-3dgs-render LANGUAGES CXX CUDA VERSION 1.1)

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

set(CC "89" CACHE STRING "Which compute capability to build for? e.g. 75, 86, 89, ...")


####################################################################
# Helper Functions
add_library(render_common
	src/dataset.cpp
	include/dataset.h
)
target_include_directories(render_common PRIVATE "${CMAKE_SOURCE_DIR}/include" "${CMAKE_SOURCE_DIR}/include/helper")
set_target_properties(render_common PROPERTIES CUDA_ARCHITECTURES OFF)
set_target_properties(render_common PROPERTIES CXX_STANDARD 17)
if (MSVC)
   target_compile_options(render_common PRIVATE /W4 $<$<CONFIG:RELEASE>:/O2> $<$<CONFIG:DEBUG>:/Od>)
else()
   target_compile_options(render_common PRIVATE -Wall -Wextra $<$<CONFIG:RELEASE>:-O3> $<$<CONFIG:DEBUG>:-O0>)
endif()

####################################################################
# CPU Implementation
add_library(render_cpu
	src/render.cpp
	include/render.h
)
target_include_directories(render_cpu PRIVATE "${CMAKE_SOURCE_DIR}/include" "${CMAKE_SOURCE_DIR}/include/helper")
set_target_properties(render_cpu PROPERTIES CUDA_ARCHITECTURES OFF)
set_target_properties(render_cpu PROPERTIES CXX_STANDARD 17)
if (MSVC)
   target_compile_options(render_cpu PRIVATE /W4 $<$<CONFIG:RELEASE>:/O2> $<$<CONFIG:DEBUG>:/Od>)
else()
   target_compile_options(render_cpu PRIVATE -Wall -Wextra $<$<CONFIG:RELEASE>:-O3> $<$<CONFIG:DEBUG>:-O0>)
endif()

####################################################################
# GPU Implementation
add_library(render_gpu
	include/render.h
	src/render.cu
)
target_include_directories(render_gpu PRIVATE "${CMAKE_SOURCE_DIR}/include" "${CMAKE_SOURCE_DIR}/include/helper")
set_target_properties(render_gpu PROPERTIES CUDA_ARCHITECTURES ${CC})
set_target_properties(render_gpu PROPERTIES CXX_STANDARD 17)
target_compile_options(render_gpu PRIVATE -Xcompiler -D_FORCE_INLINES -DVERBOSE --expt-extended-lambda -use_fast_math --expt-relaxed-constexpr -keep --ptxas-options=-v -lineinfo $<$<CONFIG:DEBUG>:-G>)


####################################################################
# Executable
add_executable(3dgs-render
	src/main.cpp
	include/render.h
)
target_include_directories(3dgs-render PRIVATE "${CMAKE_SOURCE_DIR}/include" "${CMAKE_SOURCE_DIR}/include/helper")
set_target_properties(3dgs-render PROPERTIES CUDA_ARCHITECTURES OFF)
set_target_properties(3dgs-render PROPERTIES CXX_STANDARD 17)
if (MSVC)
   target_compile_options(3dgs-render PRIVATE /W4 $<$<CONFIG:RELEASE>:/O2> $<$<CONFIG:DEBUG>:/Od>)
else()
   target_compile_options(3dgs-render PRIVATE -Wall -Wextra -Wno-missing-field-initializers $<$<CONFIG:RELEASE>:-O3> $<$<CONFIG:DEBUG>:-O0>)
endif()

# Link all together
target_link_libraries(3dgs-render render_common render_cpu render_gpu)
