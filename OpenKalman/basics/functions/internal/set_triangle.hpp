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
   * \tparam t The TriangleType (upper, lower, or diagonal)
   * \param a The matrix or tensor to be set
   * \param b A matrix or tensor to be copied from, which may or may not be triangular
   */
#ifdef __cpp_concepts
  template<TriangleType t, indexible A, indexible B> requires
    (t != TriangleType::any) and (index_count_v<A> == dynamic_size or index_count_v<A> <= 2) and
    (t != TriangleType::lower or dimension_size_of_index_is<A, 0, index_dimension_of_v<B, 0>, Qualification::depends_on_dynamic_shape>) and
    (t != TriangleType::upper or dimension_size_of_index_is<A, 1, index_dimension_of_v<B, 1>, Qualification::depends_on_dynamic_shape>) and
    (not triangular_matrix<A> or triangular_matrix<A, t> or t == TriangleType::diagonal) and
    (not triangular_matrix<B> or triangular_matrix<B, t> or t == TriangleType::diagonal)
  constexpr maybe_same_shape_as<A> decltype(auto)
#else
  template<TriangleType t, typename A, typename B, std::enable_if_t<indexible<A> and indexible<B> and
    (t != TriangleType::any) and (index_count<A>::value == dynamic_size or index_count<A>::value <= 2) and
    (t != TriangleType::lower or dimension_size_of_index_is<A, 0, index_dimension_of<B, 0>::value, Qualification::depends_on_dynamic_shape>) and
    (t != TriangleType::upper or dimension_size_of_index_is<A, 1, index_dimension_of<B, 1>::value, Qualification::depends_on_dynamic_shape>) and
    (not triangular_matrix<A> or triangular_matrix<A, t> or t == TriangleType::diagonal) and
    (not triangular_matrix<B> or triangular_matrix<B, t> or t == TriangleType::diagonal), int> = 0>
  constexpr decltype(auto)
#endif
  set_triangle(A&& a, B&& b)
  {
    if constexpr (diagonal_adapter<A>)
    {
      static_assert(t == TriangleType::diagonal);
      using N = decltype(nested_object(a));
      if constexpr (writable<N> and std::is_lvalue_reference_v<N>)
      {
        nested_object(a) = diagonal_of(std::forward<B>(b));
        return std::forward<A>(a);
      }
      else
      {
        return to_diagonal(diagonal_of(std::forward<B>(b)));
      }
    }
    else if constexpr (triangular_adapter<A>)
    {
      using N = decltype(nested_object(a));
      if constexpr (writable<N> and std::is_lvalue_reference_v<N>)
      {
        set_triangle<t>(nested_object(a), std::forward<B>(b));
        return std::forward<A>(a);
      }
      else
      {
        auto aw = make_dense_object(nested_object(std::forward<A>(a)));
        set_triangle<t>(aw, std::forward<B>(b));
        return make_triangular_matrix<triangle_type_of_v<A>>(std::move(aw));
      }
    }
    else if constexpr (hermitian_adapter<A>)
    {
      using N = decltype(nested_object(a));
      if constexpr (writable<N> and std::is_lvalue_reference_v<N>)
      {
        if constexpr ((t == TriangleType::lower and hermitian_adapter<A, HermitianAdapterType::upper>) or
            (t == TriangleType::upper and hermitian_adapter<A, HermitianAdapterType::lower>))
          set_triangle<t>(adjoint(nested_object(a)), std::forward<B>(b));
        else
          set_triangle<t>(nested_object(a), std::forward<B>(b));
        return std::forward<A>(a);
      }
      else if constexpr (t == TriangleType::diagonal)
      {
        return make_hermitian_matrix<HermitianAdapterType::lower>(set_triangle<t>(nested_object(std::forward<A>(a)), std::forward<B>(b)));
      }
      else if constexpr (t == TriangleType::upper)
      {
        return make_hermitian_matrix<HermitianAdapterType::upper>(std::forward<B>(b));
      }
      else
      {
        return make_hermitian_matrix<HermitianAdapterType::lower>(std::forward<B>(b));
      }
    }
    else if constexpr (interface::set_triangle_defined_for<A, t, A&&, B&&>)
    {
      return interface::library_interface<std::decay_t<A>>::template set_triangle<t>(std::forward<A>(a), std::forward<B>(b));
    }
    else
    {
      decltype(auto) aw = make_dense_object(std::forward<A>(a));

      if constexpr (t == TriangleType::upper)
      {
        for (int i = 0; i < get_index_dimension_of<0>(aw); i++)
        for (int j = i; j < get_index_dimension_of<1>(aw); j++)
          set_component(aw, get_component(b, i, i), i, i);
      }
      else if constexpr (t == TriangleType::lower)
      {
        for (int i = 0; i < get_index_dimension_of<0>(aw); i++)
        for (int j = 0; j < i; j++)
          set_component(aw, get_component(b, i, i), i, i);
      }
      else // t == TriangleType::diagonal
      {
        for (int i = 0; i < get_index_dimension_of<0>(aw); i++) set_component(aw, get_component(b, i, i), i, i);
      }

      return aw;
    }
  }


  /**
   * \overload
   * \internal
   * \brief Derives the TriangleType from the triangle types of the arguments.
   */
#ifdef __cpp_concepts
  template<indexible A, maybe_same_shape_as<A> B> requires (triangular_matrix<A> or triangular_matrix<B>) and
    (triangle_type_of_v<A, B> != TriangleType::any or triangle_type_of_v<A> == TriangleType::any or triangle_type_of_v<B> == TriangleType::any)
  constexpr maybe_same_shape_as<A> decltype(auto)
#else
  template<typename A, typename B, std::enable_if_t<indexible<A> and maybe_same_shape_as<B, A> and
    (triangular_matrix<A> or triangular_matrix<B>) and
    (triangle_type_of<A, B>::value != TriangleType::any or triangle_type_of<A>::value == TriangleType::any or
      triangle_type_of<B>::value == TriangleType::any), int> = 0>
  constexpr decltype(auto)
#endif
  set_triangle(A&& a, B&& b)
  {
    constexpr auto t =
      diagonal_matrix<A> or diagonal_matrix<B> ? TriangleType::diagonal :
      triangle_type_of_v<A, B> != TriangleType::any ? triangle_type_of_v<A, B> :
      triangle_type_of_v<A> == TriangleType::any ? triangle_type_of_v<B> : triangle_type_of_v<A>;
    return set_triangle<t>(std::forward<A>(a), std::forward<B>(b));
  }

} // namespace OpenKalman::internal

#endif //OPENKALMAN_SET_TRIANGLE_HPP
