/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition of \ref get_index_extent function.
 */

#ifndef OPENKALMAN_GET_INDEX_EXTENT_HPP
#define OPENKALMAN_GET_INDEX_EXTENT_HPP

#include "linear-algebra/traits/count_indices.hpp"
#include "linear-algebra/traits/index_count.hpp"
#include "linear-algebra/traits/get_mdspan.hpp"

namespace OpenKalman
{
  /**
   * \brief Get the runtime dimensions of index N of \ref indexible T
   */
#ifdef __cpp_concepts
  template<indexible T, values::index I = std::integral_constant<std::size_t, 0>>
  constexpr values::index auto
#else
  template<typename T, typename I = std::integral_constant<std::size_t, 0>,
    std::enable_if_t<indexible<T> and values::index<I>, int> = 0>
  constexpr auto
#endif
  get_index_extent(T&& t, I i = I{})
  {
    if constexpr (values::size_compares_with<I, index_count<T>, &stdex::is_gteq>)
    {
      return std::integral_constant<std::size_t, 1>{};
    }
    else if constexpr (values::size_compares_with<I, index_count<T>, &stdex::is_lt>)
    {
      constexpr auto ex = std::decay_t<decltype(get_mdspan(t))>::static_extent(i);
      if constexpr (ex == stdex::dynamic_extent)
        return get_mdspan(t).extent(static_cast<std::size_t>(i));
      else
        return std::integral_constant<std::size_t, ex>{};
    }
    else if (i < count_indices(t))
    {
      return get_mdspan(t).extent(static_cast<std::size_t>(i));
    }
    else
    {
      return 1_uz;
    }
  }


  /**
   * \overload
   */
#ifdef __cpp_concepts
  template<std::size_t I, indexible T>
  constexpr values::index auto
#else
  template<std::size_t I, typename T, std::enable_if_t<indexible<T>, int> = 0>
  constexpr auto
#endif
  get_index_extent(T&& t)
  {
    return get_index_extent(t, std::integral_constant<std::size_t, I>{});
  }

}

#endif
