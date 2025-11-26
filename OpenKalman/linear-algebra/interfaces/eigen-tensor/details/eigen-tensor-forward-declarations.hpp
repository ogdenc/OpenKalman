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
   * \tparam must_be_native T is required to be a native Eigen tensor.
   */
  template<typename T, bool must_be_native = false>
#ifdef __cpp_concepts
  concept eigen_tensor_general =
#else
  constexpr bool eigen_tensor_general =
#endif
    std::is_base_of_v<Eigen::TensorBase<std::decay_t<T>, Eigen::ReadOnlyAccessors>, std::decay_t<T>> and
      (not must_be_native or not std::is_base_of_v<EigenCustomBase, std::decay_t<T>>);


  namespace detail
  {
    template<typename T>
    struct is_eigen_tensor_wrapper : std::false_type {};

    template<typename N, typename L>
    struct is_eigen_tensor_wrapper<OpenKalman::internal::LibraryWrapper<N, L>> : std::bool_constant<eigen_tensor_general<L, true>> {};
  }


  /**
   * \internal
   * \brief T is a \ref internal::LibraryWrapper "LibraryWrapper" for T that is a tensor based on the Eigen library.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept eigen_tensor_wrapper =
#else
  constexpr bool eigen_tensor_wrapper =
#endif
  detail::is_eigen_tensor_wrapper<std::decay_t<T>>::value;


  namespace detail
  {
    template<typename>
    struct eigen_sizes;

    template<typename...Ds>
    struct eigen_sizes<std::tuple<Ds...>> { using type = Eigen::Sizes<static_cast<std::ptrdiff_t>(coordinates::dimension_of_v<Ds>)...>; };

  }


  /**
   * \internal
   * \brief Alias for the Eigen tensor version of LibraryWrapper.
   * \details A wrapper for OpenKalman classes so that they are treated exactly as native Eigen tensor types.
   * \tparam NestedObject A non-Eigen class, for which an Eigen3 trait and evaluator is defined.
   */
#ifdef __cpp_concepts
  template<indexible NestedObject>
#else
    template<typename NestedObject>
#endif
  using EigenTensorWrapper = OpenKalman::internal::LibraryWrapper<NestedObject, std::conditional_t<
    has_dynamic_dimensions<NestedObject>,
    Eigen::Tensor<
      scalar_type_of_t<NestedObject>,
      static_cast<int>(index_count_v<NestedObject>),
      layout_of_v<NestedObject> == data_layout::right ? Eigen::RowMajor : Eigen::ColMajor,
      Eigen::DenseIndex>,
    Eigen::TensorFixedSize<
      scalar_type_of_t<NestedObject>,
      typename detail::eigen_sizes<std::decay_t<decltype(all_vector_space_descriptors(std::declval<NestedObject>()))>>::type,
      layout_of_v<NestedObject> == data_layout::right ? Eigen::RowMajor : Eigen::ColMajor,
      Eigen::DenseIndex>>>;


  /**
   * \brief Trait object providing get and set routines for Eigen tensors
   */
#ifdef __cpp_concepts
  template<typename T>
  struct object_traits_tensor_base;
#else
  template<typename T, typename = void>
  struct object_traits_tensor_base;
#endif


}


#endif
