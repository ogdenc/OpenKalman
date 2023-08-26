/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Forward declarations for OpenKalman's Eigen Tensor module interface.
 */

#ifndef OPENKALMAN_EIGEN_TENSOR_FORWARD_DECLARATIONS_HPP
#define OPENKALMAN_EIGEN_TENSOR_FORWARD_DECLARATIONS_HPP

#include <type_traits>


namespace OpenKalman::Eigen3
{
  /**
   * \internal
   * \brief The ultimate base for Eigen-based tensor adapter classes in OpenKalman.
   * \details This class adds base features required by Eigen.
   */
  template<typename Derived, typename NestedMatrix>
  struct EigenTensorAdapterBase;


  /**
   * \brief Specifies any descendant of Eigen::TensorBase.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept eigen_tensor_general =
#else
  constexpr bool eigen_tensor_general =
#endif
    std::is_base_of_v<Eigen::TensorBase<std::decay_t<T>, Eigen::ReadOnlyAccessors>, std::decay_t<T>>;


  /**
   * \brief Specifies a native Eigen tensor object deriving from Eigen::TensorBase.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept native_eigen_tensor =
#else
  constexpr bool native_eigen_tensor =
#endif
    eigen_tensor_general<T> and (not std::is_base_of_v<EigenDenseBase, std::decay_t<T>>);


  /**
   * \internal
   * \brief A dumb wrapper for OpenKalman classes so that they are treated exactly as native Eigen tensor types.
   * \tparam T A non-Eigen tensor class.
   */
  template<typename T>
  struct EigenTensorWrapper;


  namespace detail
  {
    template<typename T>
    struct is_EigenTensorWrapper : std::false_type {};

    template<typename T>
    struct is_EigenTensorWrapper<EigenTensorWrapper<T>> : std::true_type {};
  }


  /**
   * \brief An instance of Eigen3::EigenTensorWrapper<T>, for any T.
   */
  template<typename T>
  #ifdef __cpp_concepts
  concept eigen_tensor_wrapper =
  #else
  constexpr bool eigen_tensor_wrapper =
  #endif
    detail::is_EigenTensorWrapper<std::decay_t<T>>::value;

} // namespace OpenKalman::Eigen3


#endif //OPENKALMAN_EIGEN_TENSOR_FORWARD_DECLARATIONS_HPP
