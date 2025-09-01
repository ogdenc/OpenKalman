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
 * \brief Definition for \ref is_vector function.
 */

#ifndef OPENKALMAN_IS_VECTOR_HPP
#define OPENKALMAN_IS_VECTOR_HPP


namespace OpenKalman
{
  namespace detail
  {
    template<std::size_t N, std::size_t...Is, typename T>
    constexpr bool get_is_vector_impl(std::index_sequence<Is...>, const T& t)
    {
      return (... and (N == Is or get_index_dimension_of<Is>(t) == 1));
    }
  }


  /**
   * \brief Return true if T is a \ref vector at runtime.
   * \details In this context, a vector is an object in which every index but one is 1D.
   * \tparam N An index designating the "large" index (e.g., 0 for a column vector, 1 for a row vector)
   * \tparam T A tensor or matrix
   * \sa vector
   */
#ifdef __cpp_concepts
  template<std::size_t N = 0, interface::count_indices_defined_for T>
#else
  template<std::size_t N = 0, typename T, std::enable_if_t<interface::count_indices_defined_for<T>, int> = 0>
#endif
  constexpr bool is_vector(const T& t)
  {
    if constexpr (values::fixed<decltype(count_indices(t))>)
    {
      constexpr std::size_t count = std::decay_t<decltype(count_indices(t))>::value;
      return detail::get_is_vector_impl<N>(std::make_index_sequence<count>{}, t);
    }
    else
    {
      for (std::size_t i = 0; i < count_indices(t); ++i) if (N != i and get_index_dimension_of(t, i) != 1) return false;
      return true;
    }
  }


}

#endif
