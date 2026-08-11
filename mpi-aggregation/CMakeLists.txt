cmake_minimum_required(VERSION 3.20)
project(mpi_alpha_agg VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ====================== macOS SDK Fix (MUST be early) ======================
if(APPLE)
    # Use the active SDK from Command Line Tools / Xcode
    execute_process(
        COMMAND xcrun --show-sdk-path
        OUTPUT_VARIABLE CMAKE_OSX_SYSROOT
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    
    # Optional but recommended: set a reasonable deployment target
    if(NOT DEFINED CMAKE_OSX_DEPLOYMENT_TARGET)
        set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "Minimum macOS deployment target")
    endif()
endif()

# ====================== Conda Environment Support ======================
if(DEFINED ENV{CONDA_PREFIX})
    list(APPEND CMAKE_PREFIX_PATH $ENV{CONDA_PREFIX})
    set(Python_ROOT_DIR $ENV{CONDA_PREFIX} CACHE PATH "")
endif()

# ====================== Find Packages ======================
find_package(MPI REQUIRED COMPONENTS C)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module NumPy)
find_package(pybind11 CONFIG REQUIRED)

# ====================== Core C Library ======================
add_library(alphaagg
    src/alphaagg_serial.c
    src/alphaagg_mpi.c
    src/alphaagg_grid.c
    src/alphaagg_dsu.c
    src/alphaagg_summary.c
)

target_include_directories(alphaagg 
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(alphaagg PUBLIC MPI::MPI_C m)

# ====================== Python Binding ======================
pybind11_add_module(_core bindings/python/alphaagg_pybind.cpp)

target_include_directories(_core PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${Python_INCLUDE_DIRS}                    # ← Added for safety
)

target_link_libraries(_core PRIVATE alphaagg MPI::MPI_C)

# ====================== Executables ======================
add_executable(alphaagg_bench apps/alphaagg_bench.c)
target_link_libraries(alphaagg_bench PRIVATE alphaagg)

add_executable(alphaagg_c_example apps/alphaagg_c_example.c)
target_link_libraries(alphaagg_c_example PRIVATE alphaagg)

set_target_properties(alphaagg_bench alphaagg_c_example 
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/apps)

set(CYTHON_AGG_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/python/mpi_alphaagg/_cython_aggregate.pyx)
set(CYTHON_AGG_C ${CMAKE_CURRENT_BINARY_DIR}/_cython_aggregate.c)
add_custom_command(
    OUTPUT ${CYTHON_AGG_C}
    COMMAND ${Python_EXECUTABLE} -m cython
            --module-name mpi_alphaagg._cython_aggregate
            --output-file ${CYTHON_AGG_C}
            ${CYTHON_AGG_SOURCE}
    DEPENDS ${CYTHON_AGG_SOURCE}
    COMMENT "Cythonizing mpi_alphaagg._cython_aggregate"
    VERBATIM)

add_library(_cython_aggregate MODULE ${CYTHON_AGG_C})
target_link_libraries(_cython_aggregate PRIVATE Python::Module Python::NumPy)
set_target_properties(_cython_aggregate PROPERTIES PREFIX "")

install(TARGETS alphaagg _core _cython_aggregate alphaagg_bench alphaagg_c_example
        LIBRARY DESTINATION mpi_alphaagg
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)
install(DIRECTORY include/mpi_alphaagg DESTINATION include)

if(BUILD_TESTING)
    add_executable(test_serial tests/test_serial.c)
    target_link_libraries(test_serial PRIVATE alphaagg)
    add_test(NAME test_serial COMMAND test_serial)

    add_executable(test_mpi_small tests/test_mpi_small.c)
    target_link_libraries(test_mpi_small PRIVATE alphaagg)

    if(MPIEXEC_EXECUTABLE)
        add_test(NAME test_mpi_small_np2
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_PREFLAGS} --oversubscribe ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:test_mpi_small>)
        add_test(NAME test_mpi_small_np4
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_PREFLAGS} --oversubscribe ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:test_mpi_small>)
    endif()
endif()