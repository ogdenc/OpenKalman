/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition of \ref get_index_pattern function.
 */

#ifndef OPENKALMAN_GET_INDEX_PATTERN_HPP
#define OPENKALMAN_GET_INDEX_PATTERN_HPP

#include "coordinates/coordinates.hpp"
#include "linear-algebra/interfaces/object-traits-defined.hpp"
#include "linear-algebra/traits/get_mdspan.hpp"
#include "linear-algebra/traits/get_pattern_collection.hpp"

namespace OpenKalman
{
  /**
   * \brief Get the \ref coordinates::pattern associated with \ref indexible object T at a given index.
   */
#ifdef __cpp_concepts
  template<indexible T, values::index I = std::integral_constant<std::size_t, 0>>
  constexpr coordinates::pattern auto
#else
  template<typename T, typename I = std::integral_constant<std::size_t, 0>,
    std::enable_if_t<indexible<T> and values::index<I>, int> = 0>
  constexpr auto
#endif
  get_index_pattern(T&& t, I i = {})
  {
    using Td = std::remove_reference_t<T>;
    if constexpr (interface::get_pattern_collection_defined_for<Td>)
    {
      return collections::get(get_pattern_collection(t), i);
    }
    else
    {
      constexpr auto ex = std::decay_t<decltype(get_mdspan(t))>::static_extent(i);
      if constexpr (ex == stdcompat::dynamic_extent)
        return get_mdspan(t).extent(static_cast<std::size_t>(i));
      else
        return std::integral_constant<std::size_t, ex>{};
    }
  }


/**
 * \overload
 */
#ifdef __cpp_concepts
  template<std::size_t I, indexible T>
  constexpr coordinates::pattern auto
#else
  template<std::size_t I, typename T, std::enable_if_t<indexible<T> and values::index<I>, int> = 0>
  constexpr auto
#endif
  get_index_pattern(T&& t)
  {
    return get_index_pattern(t, std::integral_constant<std::size_t, I>{});
  }

}

#endif
