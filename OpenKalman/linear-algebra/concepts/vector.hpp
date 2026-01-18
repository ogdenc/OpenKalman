/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2025 Christopher Lee Ogden <ogden@gatech.edu>
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

#include "linear-algebra/concepts/dimension_size_of_index_is.hpp"
#include "linear-algebra/traits/index_count.hpp"

namespace OpenKalman
{
  namespace detail
  {
    template<typename T, std::size_t N, applicability b, std::size_t...is>
    constexpr auto
    vector_fixed_index_count(std::index_sequence<is...>)
    {
      return (... and (N == is or dimension_size_of_index_is<T, is, 1, &stdex::is_eq, b>));
    }


    template<typename T, std::size_t N, applicability b>
    constexpr auto
    vector_impl()
    {
      if constexpr (not indexible<T>) // Only needed for c++17 mode
        return false;
      else
        return detail::vector_fixed_index_count<T, N, b>(std::make_index_sequence<index_count_v<T>>{});
    }

  }


  /**
   * \brief T is a vector (e.g., column or row vector).
   * \details In this context, a vector is an object in which every index but one is 1D.
   * \tparam T An indexible object
   * \tparam N An index designating the "large" index (e.g., 0 for a column vector, 1 for a row vector)
   * \tparam b Whether the vector status is guaranteed known at compile time (applicability::guaranteed), or
   * only known at runtime (applicability::permitted)
   * \sa is_vector
   */
  template<typename T, std::size_t N = 0, applicability b = applicability::guaranteed>
#ifdef __cpp_concepts
  concept vector =
#else
  constexpr bool vector =
#endif
    indexible<T> and
    detail::vector_impl<T, N, b>();


}

#endif
