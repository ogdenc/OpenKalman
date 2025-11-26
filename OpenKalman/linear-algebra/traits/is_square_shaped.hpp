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
 * \brief Definition of \ref is_square_shaped function.
 */

#ifndef OPENKALMAN_IS_SQUARE_SHAPED_HPP
#define OPENKALMAN_IS_SQUARE_SHAPED_HPP

#include <optional>
#include "coordinates/coordinates.hpp"
#include "linear-algebra/traits/count_indices.hpp"
#include "linear-algebra/traits/index_count.hpp"
#include "linear-algebra/traits/get_index_pattern.hpp"

namespace OpenKalman
{

  namespace detail
  {
    template<typename T, std::size_t i>
    constexpr std::size_t
    get_best_square_index()
    {
      if constexpr (i == 0) return i;
      else if constexpr (coordinates::fixed_pattern<decltype(get_index_pattern<i - 1>(std::declval<T>()))>) return i - 1;
      else return get_best_square_index<T, i - 1>();
    }


    template<typename T, std::size_t...is>
    constexpr auto
    is_square_shaped_impl(const T& t, std::index_sequence<is...>)
    {
      constexpr std::size_t best = get_best_square_index<T, sizeof...(is)>();
      auto best_patt = get_index_pattern<best>(t);
      using opt = std::optional<std::decay_t<decltype(best_patt)>>;
      if ((... and (is == best or coordinates::compare(get_index_pattern<is>(t), best_patt)))) return opt {best_patt};
      else return opt {};
    }
  }


  /**
   * \brief Determine whether an object is \ref square_shaped.
   * \details An object is square-shaped if it has the same extents and equivalent \ref coordinates::pattern types for each index
   * (excluding trailing 1D indices).
   * A rank-0 object is square because all indices are implicitly 1D.
   * An empty object in which the first positive n dimensions are 0 is also considered to be square.
   * \tparam T A tensor or matrix
   * \return a \ref std::optional which includes the \ref coordinates::pattern object if T is square.
   * \sa square_shaped
   */
#ifdef __cpp_concepts
  template<indexible T>
#else
  template<typename T, std::enable_if_t<indexible<T>, int> = 0>
#endif
  constexpr auto
  is_square_shaped(const T& t)
  {
    if constexpr (index_count_v<T> == stdex::dynamic_extent)
    {
      auto d0 = get_index_pattern<0>(t);
      for (std::size_t i = 1; i < count_indices(t); ++i)
      {
        if (coordinates::compare<&stdex::is_neq>(d0, get_index_pattern(t, i)))
          return std::optional<decltype((d0))> {};
      }
      return std::optional {d0};
    }
    else
    {
      constexpr std::size_t c = index_count_v<T>;
      if constexpr (c == 1)
      {
        using opt = std::optional<coordinates::Dimensions<1>>;
        if (get_index_extent<0>(t) == 1) return opt {coordinates::Dimensions<1>{}};
        else return opt {};
      }
      else
      {
        return detail::is_square_shaped_impl(t, std::make_index_sequence<c>{});
      }
    }
  }


}

#endif
