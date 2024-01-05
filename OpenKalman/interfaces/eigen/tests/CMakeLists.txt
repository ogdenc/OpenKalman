# This file is part of OpenKalman, a header-only C++ library for
# Kalman filters and other recursive filters.
#
# Created by Christopher Lee Ogden <ogden@gatech.edu> in 2017.
# Any copyright as to this file is dedicated to the Public Domain.
# https://creativecommons.org/publicdomain/zero/1.0/


cmake_minimum_required(VERSION 3.10)

if (CXX_STANDARD GREATER_EQUAL 20)
  find_package(Eigen3 3.3.9 REQUIRED CONFIG)
else()
  find_package(Eigen3 3.3 REQUIRED CONFIG)
endif()

enable_testing()
find_package(GTest REQUIRED)
include(GoogleTest)

add_executable(eigen3
        traits-tests/Array.test.cpp
        traits-tests/ArrayWrapper.test.cpp
        traits-tests/Block.test.cpp
        traits-tests/CwiseBinaryOp.test.cpp
        traits-tests/CwiseNullaryOp.test.cpp
        traits-tests/CwiseTernaryOp.test.cpp
        traits-tests/CwiseUnaryOp.test.cpp
        traits-tests/CwiseUnaryView.test.cpp
        traits-tests/Diagonal.test.cpp
        traits-tests/DiagonalMatrix.test.cpp
        traits-tests/DiagonalWrapper.test.cpp
        traits-tests/Homogeneous.test.cpp
        traits-tests/IndexedView.test.cpp
        traits-tests/Inverse.test.cpp
        traits-tests/Map.test.cpp
        traits-tests/Matrix.test.cpp
        traits-tests/MatrixWrapper.test.cpp
        traits-tests/PartialReduxExpr.test.cpp
        traits-tests/PermutationWrapper.test.cpp
        traits-tests/Product.test.cpp
        traits-tests/Ref.test.cpp
        traits-tests/Replicate.test.cpp
        traits-tests/Reshaped.test.cpp
        traits-tests/Reverse.test.cpp
        traits-tests/Select.test.cpp
        traits-tests/SelfAdjointView.test.cpp
        traits-tests/Solve.test.cpp
        traits-tests/Transpose.test.cpp
        traits-tests/TriangularView.test.cpp
        traits-tests/VectorBlock.test.cpp
        traits-tests/VectorWiseOp.test.cpp

        LibraryWrapper.test.cpp
        FixedSizeAdapter.test.cpp

        constant-objects.test.cpp
        constant-diagonal-objects.test.cpp
        shape-traits.test.cpp
        special-matrix-traits.test.cpp

        function-tests/property-functions.test.cpp
        function-tests/make-functions.test.cpp

        function-tests/eigen-diagonal.test.cpp
        function-tests/eigen-nary-operations.test.cpp
        function-tests/eigen-reductions.test.cpp
        function-tests/eigen-blocks.test.cpp
        function-tests/eigen-chipwise-operations.test.cpp

        function-tests/transpose-adjoint-conjugate.test.cpp
        function-tests/trace.test.cpp
        function-tests/determinant.test.cpp
        function-tests/sum.test.cpp

        function-tests/eigen-contract.test.cpp
        function-tests/eigen-rank-update.test.cpp
        function-tests/eigen-solve.test.cpp
        function-tests/eigen-decomposition.test.cpp
    )

target_link_libraries(eigen3 ${GTEST_BOTH_LIBRARIES} Eigen3::Eigen Threads::Threads)

gtest_discover_tests(eigen3 DISCOVERY_TIMEOUT 20)
