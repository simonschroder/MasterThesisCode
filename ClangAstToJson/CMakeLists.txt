
cmake_minimum_required(VERSION 3.13)
project(ClangAstToJson)

# Non-Cmake command:
#g++ -std=c++14 lala4.cpp -lclangRewriteFrontend -lclangTooling -lclangRewrite -lclangFrontend -lclangSerialization -lclangDriver -lclangParse -lclangSema -lclangAnalysis -lclangEdit -lclangAST -lclangLex -lclangBasic -lLLVMAsmParser -lLLVMMCParser -lLLVMBitReader -lLLVMTransformUtils -lLLVMCore -lLLVMMC -lLLVMOption -lLLVMSupport $(llvm-config-7 --cxxflags --ldflags --libs --system-libs --link-static)

# Use C++14
set(CMAKE_CXX_STANDARD 14)

# Uncomment for verbose output:
# set(CMAKE_VERBOSE_MAKEFILE ON)

# TODO: Uncommenting the following should enable static linkage but linkage does not work then
#set(LINK_STATIC --link-static)
#SET(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
#set(BUILD_SHARED_LIBS OFF)
#target_link_libraries(ClangAstToJson -static)

# llvm-config which associated LLVM/Clang should be used:
set(LLVM_CONF llvm-config-7)

# Determine LLVM info:
execute_process(COMMAND ${LLVM_CONF} --src-root OUTPUT_VARIABLE LLVM_PATH OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${LLVM_CONF} ${LINK_STATIC} --system-libs OUTPUT_VARIABLE LLVM_SYSTEM_LIBS OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${LLVM_CONF} ${LINK_STATIC} --libs OUTPUT_VARIABLE LLVM_LIBS OUTPUT_STRIP_TRAILING_WHITESPACE)
#execute_process(COMMAND ${LLVM_CONF} --ldflags OUTPUT_VARIABLE LLVM_LDFLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
message(STATUS "Using LLVM in: ${LLVM_PATH}")
message(STATUS "LLVM_SYSTEM_LIBS: ${LLVM_SYSTEM_LIBS}")
message(STATUS "LLVM_LIBS: ${LLVM_LIBS}")
#message(STATUS "LLVM_LDFLAGS: ${LLVM_LDFLAGS}")

# TODO: Use paths from llvm-config?
link_directories(${LLVM_PATH}/lib)
include_directories(${LLVM_PATH}/include)

# Add binary to compile:
add_executable(ClangAstToJson main.cpp)

# Link against LLVM libs:
target_link_libraries(ClangAstToJson ${LLVM_LIBS})
target_link_libraries(ClangAstToJson ${LLVM_SYSTEM_LIBS})

# Link against Clang libs:
target_link_libraries(ClangAstToJson
        clangRewriteFrontend
        clangTooling
        clangRewrite
        clangFrontend
        clangSerialization
        clangDriver
        clangParse
        clangSema
        clangAnalysis
        clangEdit
        clangAST
        clangLex
        clangBasic
        LLVMAsmParser
        LLVMMCParser
        LLVMBitReader
        LLVMTransformUtils
        LLVMCore
        LLVMMC
        LLVMOption
        LLVMSupport)
