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
 * \brief Definition for \ref one_dimensional.
 */

#ifndef OPENKALMAN_ONE_DIMENSIONAL_HPP
#define OPENKALMAN_ONE_DIMENSIONAL_HPP

#include "linear-algebra/traits/index_count.hpp"
#include "linear-algebra/concepts/dimension_size_of_index_is.hpp"

namespace OpenKalman
{
  namespace detail
  {
    template<typename T, applicability b, std::size_t...is>
    constexpr auto
    one_dimensional_fixed_index_count(std::index_sequence<is...>)
    {
      return (... and (dimension_size_of_index_is<T, is, 1, &stdex::is_eq, b>));
    }


    template<typename T, applicability b>
    constexpr auto
    one_dimensional_impl()
    {
      if constexpr (not indexible<T>) // only needed in c++17 mode
        return false;
      else if constexpr (index_count_v<T> == stdex::dynamic_extent)
        return b == applicability::permitted;
      else
        return detail::one_dimensional_fixed_index_count<T, b>(std::make_index_sequence<index_count_v<T>>{});
    }
  }


  /**
   * \brief Specifies that a type is one-dimensional in every index.
   * \details Each index also must have an equivalent \ref coordinates::pattern object.
   */
  template<typename T, applicability b = applicability::guaranteed>
#ifdef __cpp_concepts
  concept one_dimensional =
#else
  constexpr inline bool one_dimensional =
#endif
    indexible<T> and
    detail::one_dimensional_impl<T, b>();


}

#endif
