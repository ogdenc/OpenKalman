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
 * \brief Definition of \ref eigen_wrapper and EigenWrapper alias.
 */

#ifndef OPENKALMAN_EIGEN_WRAPPER_HPP
#define OPENKALMAN_EIGEN_WRAPPER_HPP

namespace OpenKalman::Eigen3
{

  namespace detail
  {
    template<typename T>
    struct is_eigen_wrapper : std::false_type {};

    template<typename N, typename L>
    struct is_eigen_wrapper<internal::LibraryWrapper<N, L>> : std::bool_constant<eigen_general<L, true>> {};
  } // namespace detail


  /**
   * \internal
   * \brief T is a \ref internal::LibraryWrapper "LibraryWrapper" for T based on the Eigen library.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept eigen_wrapper =
#else
  constexpr bool eigen_wrapper =
#endif
    detail::is_eigen_wrapper<std::decay_t<T>>::value;


  /**
   * \internal
   * \brief Alias for the Eigen version of LibraryWrapper.
   * \details A wrapper for OpenKalman classes so that they are treated exactly as native Eigen types.
   * \tparam NestedObject A non-Eigen class, for which an Eigen3 trait and evaluator is defined.
   */
#ifdef __cpp_concepts
  template<indexible NestedObject> requires (index_count_v<NestedObject> <= 2)
#else
  template<typename NestedObject>
#endif
  using EigenWrapper = internal::LibraryWrapper<NestedObject,
    std::conditional_t<eigen_array_general<NestedObject>,
      Eigen::Array<
        scalar_type_of_t<NestedObject>,
        dynamic_dimension<NestedObject, 0> ? Eigen::Dynamic : static_cast<int>(index_dimension_of_v<NestedObject, 0>),
        dynamic_dimension<NestedObject, 1> ? Eigen::Dynamic : static_cast<int>(index_dimension_of_v<NestedObject, 1>),
        layout_of_v<NestedObject> == Layout::right ? Eigen::RowMajor : Eigen::ColMajor>,
      Eigen::Matrix<
        scalar_type_of_t<NestedObject>,
        dynamic_dimension<NestedObject, 0> ? Eigen::Dynamic : static_cast<int>(index_dimension_of_v<NestedObject, 0>),
        dynamic_dimension<NestedObject, 1> ? Eigen::Dynamic : static_cast<int>(index_dimension_of_v<NestedObject, 1>),
        layout_of_v<NestedObject> == Layout::right ? Eigen::RowMajor : Eigen::ColMajor>>>;


  /**
   * Make a \ref LibraryWrapper for the Eigen library.
   */
#ifdef __cpp_concepts
  template<indexible Arg> requires (index_count_v<Arg> <= 2)
#else
  template<typename Arg, std::enable_if_t<indexible<Arg> and (index_count_v<Arg> <= 2), int> = 0>
#endif
  inline auto
  make_eigen_wrapper(Arg&& arg)
  {
    return EigenWrapper<Arg> {std::forward<Arg>(arg)};
  }

} // namespace OpenKalman::Eigen3


#endif //OPENKALMAN_EIGEN_WRAPPER_HPP
