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

}


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
  struct traits<OpenKalman::constant_adapter<PatternMatrix, Scalar, constant...>>
    : OpenKalman::Eigen3::internal::native_traits<OpenKalman::constant_adapter<PatternMatrix, Scalar, constant...>> {};


  template<typename NestedObject, OpenKalman::HermitianAdapterType storage_triangle>
  struct traits<OpenKalman::HermitianAdapter<NestedObject, storage_triangle>>
    : OpenKalman::Eigen3::internal::native_traits<OpenKalman::HermitianAdapter<NestedObject, storage_triangle>> {};


  template<typename NestedObject, OpenKalman::triangle_type tri>
  struct traits<OpenKalman::TriangularAdapter<NestedObject, tri>>
    : OpenKalman::Eigen3::internal::native_traits<OpenKalman::TriangularAdapter<NestedObject, tri>> {};


  template<typename NestedObject>
  struct traits<OpenKalman::diagonal_adapter<NestedObject>>
    : OpenKalman::Eigen3::internal::native_traits<OpenKalman::diagonal_adapter<NestedObject>> {};


  template<typename NestedObject>
  struct traits<OpenKalman::ToEuclideanExpr<NestedObject>>
    : OpenKalman::Eigen3::internal::native_traits<OpenKalman::ToEuclideanExpr<NestedObject>> {};


  template<typename NestedObject, typename V0>
  struct traits<OpenKalman::FromEuclideanExpr<NestedObject, V0>>
    : OpenKalman::Eigen3::internal::native_traits<OpenKalman::FromEuclideanExpr<NestedObject, V0>> {};

  
  
  
  
  

  template<typename StaticDescriptor, typename NestedObject>
  struct traits<OpenKalman::Mean<StaticDescriptor, NestedObject>>
    : OpenKalman::Eigen3::internal::native_traits<OpenKalman::Mean<StaticDescriptor, NestedObject>> {};


  template<typename RowCoefficients, typename ColumnCoefficients, typename ArgType>
  struct traits<OpenKalman::Matrix<RowCoefficients, ColumnCoefficients, ArgType>>
    : traits<std::decay_t<ArgType>> {};


  template<typename Coeffs, typename NestedObject>
  struct traits<OpenKalman::EuclideanMean<Coeffs, NestedObject>>
    : traits<std::decay_t<NestedObject>> {};


  template<typename StaticDescriptor, typename NestedObject>
  struct traits<OpenKalman::Covariance<StaticDescriptor, NestedObject>>
    : OpenKalman::Eigen3::internal::native_traits<OpenKalman::Covariance<StaticDescriptor, NestedObject>> {};


  template<typename StaticDescriptor, typename NestedObject>
  struct traits<OpenKalman::SquareRootCovariance<StaticDescriptor, NestedObject>>
    : OpenKalman::Eigen3::internal::native_traits<OpenKalman::SquareRootCovariance<StaticDescriptor, NestedObject>> {};


}

#endif
