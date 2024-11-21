/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref vector.
 */

#ifndef OPENKALMAN_VECTOR_HPP
#define OPENKALMAN_VECTOR_HPP


namespace OpenKalman
{
  namespace detail
  {
    template<typename T, std::size_t N, Qualification b, std::size_t...Is>
    constexpr bool do_vector_impl(std::index_sequence<Is...>)
    {
      return (... and (N == Is or (b == Qualification::depends_on_dynamic_shape and dynamic_dimension<T, Is>) or dimension_size_of_index_is<T, Is, 1>));
    }


    // If index_count<T> is dynamic, at least check indices until N + 1.
#ifdef __cpp_concepts
    template<typename T, std::size_t N, Qualification b>
#else
    template<typename T, std::size_t N, Qualification b, typename = void>
#endif
    struct vector_impl : std::bool_constant<
      b == Qualification::depends_on_dynamic_shape and detail::do_vector_impl<T, N, b>(std::make_index_sequence<N + 1> {})> {};

    // If index_count<T> is static, check all indices.
#ifdef __cpp_concepts
    template<typename T, std::size_t N, Qualification b> requires (index_count_v<T> != dynamic_size)
    struct vector_impl<T, N, b>
#else
    template<typename T, std::size_t N, Qualification b>
    struct vector_impl<T, N, b, std::enable_if_t<index_count<T>::value != dynamic_size>>
#endif
      : std::bool_constant<detail::do_vector_impl<T, N, b>(std::make_index_sequence<index_count_v<T>> {})> {};


  } // namespace detail


  /**
   * \brief T is a vector (e.g., column or row vector).
   * \details In this context, a vector is an object in which every index but one is 1D.
   * \tparam T An indexible object
   * \tparam N An index designating the "large" index (0 for a column vector, 1 for a row vector)
   * \tparam b Whether the vector status is unqualified known at compile time (Qualification::unqualified), or
   * only known at runtime (Qualification::depends_on_dynamic_shape)
   * \sa is_vector
   */
  template<typename T, std::size_t N = 0, Qualification b = Qualification::unqualified>
#ifdef __cpp_concepts
  concept vector =
#else
  constexpr bool vector =
#endif
    indexible<T> and detail::vector_impl<T, N, b>::value;


} // namespace OpenKalman

#endif //OPENKALMAN_VECTOR_HPP
