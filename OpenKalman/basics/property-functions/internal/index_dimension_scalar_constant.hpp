/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \internal
 * \brief Definition for indexible_dimension_scalar_constant_of function.
 */

#ifndef OPENKALMAN_INDEX_DIMENSION_SCALAR_CONSTANT_HPP
#define OPENKALMAN_INDEX_DIMENSION_SCALAR_CONSTANT_HPP

namespace OpenKalman::internal
{
  /**
   * \internal
   * \brief Returns a scalar constant reflecting the size of an index for a tensor or matrix.
   * \details The return value is a known or unknown \ref scalar_constant of the same scalar type as T.
   * \tparam N The index
   * \tparam T The matrix, expression, or array
   * \internal \sa interface::indexible_object_traits
   */
#ifdef __cpp_concepts
  template<std::size_t N, interface::count_indices_defined_for T> requires
    interface::get_vector_space_descriptor_defined_for<T> and interface::scalar_type_defined_for<T>
#else
  template<std::size_t N, typename T, std::enable_if_t<interface::count_indices_defined_for<T> and
    interface::get_vector_space_descriptor_defined_for<T> and interface::scalar_type_defined_for<T>, int> = 0>
#endif
  constexpr auto index_dimension_scalar_constant(const T& t)
  {
    using Scalar = typename interface::indexible_object_traits<std::decay_t<T>>::scalar_type;
    if constexpr (static_index_value<decltype(get_index_dimension_of<N>(t))>)
    {
      constexpr std::size_t I = std::decay_t<decltype(get_index_dimension_of<N>(t))>::value;
      return ScalarConstant<Scalar, I>{};
    }
    else
    {
      return static_cast<Scalar>(get_index_dimension_of<N>(t));
    }
  }


  /**
   * \overload
   * \internal
   * \brief Returns a scalar constant reflecting the size of an index for a tensor or matrix.
   * \details The return value is a known or unknown \ref scalar_constant of the same scalar type as T.
   * \tparam T The matrix, expression, or array
   * \tparam N An \ref index_value
   * \internal \sa interface::indexible_object_traits
   */
#ifdef __cpp_concepts
  template<interface::count_indices_defined_for T, index_value N> requires
    interface::get_vector_space_descriptor_defined_for<T> and interface::scalar_type_defined_for<T>
#else
  template<typename T, typename N, std::enable_if_t<interface::count_indices_defined_for<T> and index_value<N> and
    interface::get_vector_space_descriptor_defined_for<T> and interface::scalar_type_defined_for<T>, int> = 0>
#endif
  constexpr auto index_dimension_scalar_constant(const T& t, const N& n)
  {
    if constexpr (static_index_value<N>) return index_dimension_scalar_constant<N::value>(t);
    else return static_cast<typename interface::indexible_object_traits<std::decay_t<T>>::scalar_type>(get_index_dimension_of(t, n));
  }


} // namespace OpenKalman::internal

#endif //OPENKALMAN_INDEX_DIMENSION_SCALAR_CONSTANT_HPP
