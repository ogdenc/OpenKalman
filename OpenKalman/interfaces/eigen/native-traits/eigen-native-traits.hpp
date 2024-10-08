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
  template<typename NestedObject>
#else
  template<typename NestedObject, typename = void>
#endif
  struct native_traits;

} // namespace OpenKalman::Eigen3::internal


namespace Eigen::internal
{
  template<typename T, typename L>
  struct traits<OpenKalman::internal::LibraryWrapper<T, L>>
    : OpenKalman::Eigen3::internal::native_traits<OpenKalman::internal::LibraryWrapper<T, L>> {};


  template<typename NestedObject, typename...Vs>
  struct traits<OpenKalman::internal::FixedSizeAdapter<NestedObject, Vs...>>
    : OpenKalman::Eigen3::internal::native_traits<OpenKalman::internal::FixedSizeAdapter<NestedObject, Vs...>> {};


  template<typename NestedObject, typename...Vs>
  struct traits<OpenKalman::VectorSpaceAdapter<NestedObject, Vs...>>
    : OpenKalman::Eigen3::internal::native_traits<OpenKalman::VectorSpaceAdapter<NestedObject, Vs...>> {};


  template<typename PatternMatrix, typename Scalar, auto...constant>
  struct traits<OpenKalman::ConstantAdapter<PatternMatrix, Scalar, constant...>>
    : OpenKalman::Eigen3::internal::native_traits<OpenKalman::ConstantAdapter<PatternMatrix, Scalar, constant...>> {};


  template<typename NestedObject, OpenKalman::HermitianAdapterType storage_triangle>
  struct traits<OpenKalman::SelfAdjointMatrix<NestedObject, storage_triangle>>
    : OpenKalman::Eigen3::internal::native_traits<OpenKalman::SelfAdjointMatrix<NestedObject, storage_triangle>> {};


  template<typename NestedObject, OpenKalman::TriangleType triangle_type>
  struct traits<OpenKalman::TriangularMatrix<NestedObject, triangle_type>>
    : OpenKalman::Eigen3::internal::native_traits<OpenKalman::TriangularMatrix<NestedObject, triangle_type>> {};


  template<typename NestedObject>
  struct traits<OpenKalman::DiagonalMatrix<NestedObject>>
    : OpenKalman::Eigen3::internal::native_traits<OpenKalman::DiagonalMatrix<NestedObject>> {};


  template<typename Coeffs, typename NestedObject>
  struct traits<OpenKalman::ToEuclideanExpr<Coeffs, NestedObject>>
    : OpenKalman::Eigen3::internal::native_traits<OpenKalman::ToEuclideanExpr<Coeffs, NestedObject>> {};


  template<typename Coeffs, typename NestedObject>
  struct traits<OpenKalman::FromEuclideanExpr<Coeffs, NestedObject>>
    : OpenKalman::Eigen3::internal::native_traits<OpenKalman::FromEuclideanExpr<Coeffs, NestedObject>> {};

  
  
  
  
  

  template<typename FixedDescriptor, typename NestedObject>
  struct traits<OpenKalman::Mean<FixedDescriptor, NestedObject>>
    : OpenKalman::Eigen3::internal::native_traits<OpenKalman::Mean<FixedDescriptor, NestedObject>> {};


  template<typename RowCoefficients, typename ColumnCoefficients, typename ArgType>
  struct traits<OpenKalman::Matrix<RowCoefficients, ColumnCoefficients, ArgType>>
    : traits<std::decay_t<ArgType>> {};


  template<typename Coeffs, typename NestedObject>
  struct traits<OpenKalman::EuclideanMean<Coeffs, NestedObject>>
    : traits<std::decay_t<NestedObject>> {};


  template<typename FixedDescriptor, typename NestedObject>
  struct traits<OpenKalman::Covariance<FixedDescriptor, NestedObject>>
    : OpenKalman::Eigen3::internal::native_traits<OpenKalman::Covariance<FixedDescriptor, NestedObject>> {};


  template<typename FixedDescriptor, typename NestedObject>
  struct traits<OpenKalman::SquareRootCovariance<FixedDescriptor, NestedObject>>
    : OpenKalman::Eigen3::internal::native_traits<OpenKalman::SquareRootCovariance<FixedDescriptor, NestedObject>> {};


}

#endif //OPENKALMAN_EIGEN_NATIVE_TRAITS_HPP
