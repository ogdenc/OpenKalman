/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \dir
 * \brief Files relating to the interface to the Eigen3 library.
 *
 * \file
 * \brief Header file for traits for Eigen3 classes.
 */

#ifndef OPENKALMAN_EIGEN_TRAITS_HPP
#define OPENKALMAN_EIGEN_TRAITS_HPP


#include "indexible_object_traits_base.hpp"
#include "eigen-library-interface.hpp"

#include "linear-algebra/interfaces/eigen/traits/functors/nullary.hpp"
#include "linear-algebra/interfaces/eigen/traits/functors/unary.hpp"
#include "linear-algebra/interfaces/eigen/traits/functors/binary.hpp"
#include "linear-algebra/interfaces/eigen/traits/functors/ternary.hpp"
#include "linear-algebra/interfaces/eigen/traits/functors/redux.hpp"
#include "linear-algebra/interfaces/eigen/traits/functors/functor_composition.hpp"

#include "Array.hpp"
#include "ArrayWrapper.hpp"
#include "Block.hpp"
#include "CwiseBinaryOp.hpp"
#include "CwiseNullaryOp.hpp"
#include "CwiseUnaryOp.hpp"
#include "CwiseUnaryView.hpp"
#include "CwiseTernaryOp.hpp"
#include "Diagonal.hpp"
#include "DiagonalMatrix.hpp"
#include "DiagonalWrapper.hpp"
#include "Homogeneous.hpp"

#if EIGEN_VERSION_AT_LEAST(3,4,0)
#include "IndexedView.hpp"
#endif

#include "Inverse.hpp"
#include "Map.hpp"
#include "Matrix.hpp"
#include "MatrixWrapper.hpp"
#include "NestByValue.hpp"
#include "PermutationMatrix.hpp"
#include "PermutationWrapper.hpp"
#include "Product.hpp"
#include "Ref.hpp"
#include "Replicate.hpp"

#if EIGEN_VERSION_AT_LEAST(3,4,0)
#include "Reshaped.hpp"
#endif

#include "Reverse.hpp"
#include "Select.hpp"
#include "SelfAdjointView.hpp"
#include "Solve.hpp"
#include "Transpose.hpp"
#include "TriangularView.hpp"
#include "VectorBlock.hpp"
#include "VectorWiseOp.hpp"

#include "PartialReduxExpr.hpp"


#endif //OPENKALMAN_EIGEN_TRAITS_HPP
