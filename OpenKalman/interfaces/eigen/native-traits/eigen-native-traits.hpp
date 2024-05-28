/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \internal
 * \file
 * \brief Native Eigen3 traits for all Eigen3 extensions, including tensors
 */

#ifndef OPENKALMAN_EIGEN_NATIVE_TRAITS_HPP
#define OPENKALMAN_EIGEN_NATIVE_TRAITS_HPP

namespace OpenKalman::Eigen3::internal
{
#ifdef __cpp_concepts
  template<typename NestedMatrix>
#else
  template<typename NestedMatrix, typename = void>
#endif
  struct native_traits;

} // namespace OpenKalman::Eigen3::internal


namespace Eigen::internal
{
  template<typename T, typename L>
  struct traits<OpenKalman::internal::LibraryWrapper<T, L>>
    : OpenKalman::Eigen3::internal::native_traits<OpenKalman::internal::LibraryWrapper<T, L>> {};


  template<typename T, typename...Ps>
  struct traits<OpenKalman::internal::SelfContainedWrapper<T, Ps...>>
    : OpenKalman::Eigen3::internal::native_traits<OpenKalman::internal::SelfContainedWrapper<T, Ps...>> {};


  template<typename NestedMatrix, typename...Vs>
  struct traits<OpenKalman::internal::FixedSizeAdapter<NestedMatrix, Vs...>>
    : OpenKalman::Eigen3::internal::native_traits<OpenKalman::internal::FixedSizeAdapter<NestedMatrix, Vs...>> {};


  template<typename PatternMatrix, typename Scalar, auto...constant>
  struct traits<OpenKalman::ConstantAdapter<PatternMatrix, Scalar, constant...>>
    : OpenKalman::Eigen3::internal::native_traits<OpenKalman::ConstantAdapter<PatternMatrix, Scalar, constant...>> {};


  template<typename NestedMatrix, OpenKalman::HermitianAdapterType storage_triangle>
  struct traits<OpenKalman::SelfAdjointMatrix<NestedMatrix, storage_triangle>>
    : OpenKalman::Eigen3::internal::native_traits<OpenKalman::SelfAdjointMatrix<NestedMatrix, storage_triangle>> {};


  template<typename NestedMatrix, OpenKalman::TriangleType triangle_type>
  struct traits<OpenKalman::TriangularMatrix<NestedMatrix, triangle_type>>
    : OpenKalman::Eigen3::internal::native_traits<OpenKalman::TriangularMatrix<NestedMatrix, triangle_type>> {};


  template<typename NestedMatrix>
  struct traits<OpenKalman::DiagonalMatrix<NestedMatrix>>
    : OpenKalman::Eigen3::internal::native_traits<OpenKalman::DiagonalMatrix<NestedMatrix>> {};


  template<typename Coeffs, typename NestedMatrix>
  struct traits<OpenKalman::ToEuclideanExpr<Coeffs, NestedMatrix>>
    : OpenKalman::Eigen3::internal::native_traits<OpenKalman::ToEuclideanExpr<Coeffs, NestedMatrix>> {};


  template<typename Coeffs, typename NestedMatrix>
  struct traits<OpenKalman::FromEuclideanExpr<Coeffs, NestedMatrix>>
    : OpenKalman::Eigen3::internal::native_traits<OpenKalman::FromEuclideanExpr<Coeffs, NestedMatrix>> {};


  template<typename FixedDescriptor, typename NestedMatrix>
  struct traits<OpenKalman::Mean<FixedDescriptor, NestedMatrix>>
    : OpenKalman::Eigen3::internal::native_traits<OpenKalman::Mean<FixedDescriptor, NestedMatrix>> {};


  template<typename RowCoefficients, typename ColumnCoefficients, typename ArgType>
  struct traits<OpenKalman::Matrix<RowCoefficients, ColumnCoefficients, ArgType>>
    : traits<std::decay_t<ArgType>> {};


  template<typename Coeffs, typename NestedMatrix>
  struct traits<OpenKalman::EuclideanMean<Coeffs, NestedMatrix>>
    : traits<std::decay_t<NestedMatrix>> {};


  template<typename FixedDescriptor, typename NestedMatrix>
  struct traits<OpenKalman::Covariance<FixedDescriptor, NestedMatrix>>
    : OpenKalman::Eigen3::internal::native_traits<OpenKalman::Covariance<FixedDescriptor, NestedMatrix>> {};


  template<typename FixedDescriptor, typename NestedMatrix>
  struct traits<OpenKalman::SquareRootCovariance<FixedDescriptor, NestedMatrix>>
    : OpenKalman::Eigen3::internal::native_traits<OpenKalman::SquareRootCovariance<FixedDescriptor, NestedMatrix>> {};


}

#endif //OPENKALMAN_EIGEN_NATIVE_TRAITS_HPP
