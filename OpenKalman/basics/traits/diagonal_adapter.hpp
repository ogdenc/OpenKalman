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
 * \file
 * \brief Definition for \ref diagonal_adapter.
 */

#ifndef OPENKALMAN_DIAGONAL_ADAPTER_HPP
#define OPENKALMAN_DIAGONAL_ADAPTER_HPP


namespace OpenKalman
{
#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, std::size_t N, typename = void>
    struct nested_is_vector : std::false_type {};

    template<typename T, std::size_t N>
    struct nested_is_vector<T, N, std::enable_if_t<has_nested_object<T>>>
      : std::bool_constant<vector<nested_object_of_t<T>, N>> {};
  } // namespace detail
#endif


  /**
   * \brief Specifies that a type is a diagonal matrix adapter.
   * \details This is an adapter that takes a \ref vector and produces a \ref diagonal_matrix.
   * Components outside the diagonal are zero.
   * \tparam T A matrix or tensor.
   * \tparam N An index designating the "large" index of the vector (0 for a column vector, 1 for a row vector)
   */
  template<typename T, std::size_t N = 0>
#ifdef __cpp_concepts
  concept diagonal_adapter = interface::indexible_object_traits<std::decay_t<T>>::template is_triangular<TriangleType::diagonal> and
    vector<nested_object_of_t<T>, N>;
#else
  constexpr bool diagonal_adapter = internal::is_explicitly_triangular<T, TriangleType::diagonal>::value and detail::nested_is_vector<T, N>::value;
#endif


} // namespace OpenKalman

#endif //OPENKALMAN_DIAGONAL_ADAPTER_HPP
