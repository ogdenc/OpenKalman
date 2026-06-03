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
 * \brief Definition for set_triangle function.
 */

#ifndef OPENKALMAN_SET_TRIANGLE_HPP
#define OPENKALMAN_SET_TRIANGLE_HPP

namespace OpenKalman::internal
{
  /**
   * \internal
   * \brief Set only a triangular (upper or lower) or diagonal part of a matrix by copying from another matrix.
   * \details If a can be modified, this will likely result in an in-place update within a.
   * To determine whether this occurred, check whether the result is a reference to a. For example, for result type R,
   * check whether <code>std::is_reference_v<R></code> and <code>std::is_same_v<std::decay_t<R>, std::decay_t<A>></code>,
   * \tparam t The triangle_type (upper, lower, or diagonal)
   * \param a The matrix or tensor to be set
   * \param b A matrix or tensor to be copied from, which may or may not be triangular
   * \returns Reference to a, as modified.
   */
#ifdef __cpp_concepts
  template<triangle_type t, indexible A, indexible B> requires
    (index_count_v<A> == stdex::dynamic_extent or index_count_v<A> <= 2) and
    (t != triangle_type::lower or dimension_size_of_index_is<A, 0, index_dimension_of_v<B, 0>, applicability::permitted>) and
    (t != triangle_type::upper or dimension_size_of_index_is<A, 1, index_dimension_of_v<B, 1>, applicability::permitted>) and
    (t != triangle_type::lower or not triangular_matrix<B, triangle_type::upper>) and
    (t != triangle_type::upper or not triangular_matrix<B, triangle_type::lower>)
#else
  template<triangle_type t, typename A, typename B, std::enable_if_t<indexible<A> and indexible<B> and
    (index_count<A>::value == stdex::dynamic_extent or index_count<A>::value <= 2) and
    (t != triangle_type::lower or dimension_size_of_index_is<A, 0, index_dimension_of<B, 0>::value, applicability::permitted>) and
    (t != triangle_type::upper or dimension_size_of_index_is<A, 1, index_dimension_of<B, 1>::value, applicability::permitted>) and
    (t != triangle_type::lower or not triangular_matrix<B, triangle_type::upper>) and
    (t != triangle_type::upper or not triangular_matrix<B, triangle_type::lower>), int> = 0>
#endif
  constexpr A&&
  set_triangle(A&& a, B&& b)
  {
    if constexpr (interface::set_triangle_defined_for<A, t, A&&, B&&>)
    {
      interface::library_interface<stdex::remove_cvref_t<A>>::template set_triangle<t>(std::forward<A>(a), std::forward<B>(b));
    }
    else if constexpr (interface::set_triangle_defined_for<A, t, A&&, decltype(to_native_matrix<A>(std::declval<B&&>()))>)
    {
      interface::library_interface<stdex::remove_cvref_t<A>>::template set_triangle<t>(std::forward<A>(a), to_native_matrix<A>(std::forward<B>(b)));
    }
    else if constexpr (triangular_adapter<A>)
    {
      set_triangle<t>(nested_object(a), std::forward<B>(b));
    }
    else if constexpr (hermitian_adapter_concept<A>)
    {
      if constexpr ((t == triangle_type::lower and hermitian_adapter_concept<A, triangle_type::upper>) or
          (t == triangle_type::upper and hermitian_adapter_concept<A, triangle_type::lower>))
        set_triangle<t>(nested_object(a), conjugate_transpose(std::forward<B>(b)));
      else
        set_triangle<t>(nested_object(a), std::forward<B>(b));
    }
    else if constexpr (diagonal_matrix<A> and internal::has_nested_vector<A, 0>)
    {
      copy(nested_object(a), diagonal_of(std::forward<B>(b)));
    }
    else if constexpr (diagonal_matrix<A> and internal::has_nested_vector<A, 1>)
    {
      copy(nested_object(a), transpose(diagonal_of(std::forward<B>(b))));
    }
    else if constexpr (t == triangle_type::upper)
    {
      for (int i = 0; i < get_index_dimension_of<0>(a); i++)
      for (int j = i; j < get_index_dimension_of<1>(a); j++)
        set_component(a, get_component(b, i, i), i, i);
    }
    else if constexpr (t == triangle_type::lower)
    {
      for (int i = 0; i < get_index_dimension_of<0>(a); i++)
      for (int j = 0; j < i; j++)
        set_component(a, get_component(b, i, i), i, i);
    }
    else // t == triangle_type::diagonal
    {
      for (int i = 0; i < get_index_dimension_of<0>(a); i++) set_component(a, get_component(b, i, i), i, i);
    }
    return std::forward<A>(a);
  }


}

#endif
