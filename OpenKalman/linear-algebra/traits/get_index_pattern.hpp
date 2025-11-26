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
#include "linear-algebra/interfaces/interfaces-defined.hpp"
#include "linear-algebra/traits/count_indices.hpp"
#include "linear-algebra/traits/index_count.hpp"
#include "linear-algebra/traits/get_mdspan.hpp"
#include "linear-algebra/traits/get_pattern_collection.hpp"

namespace OpenKalman
{
  /**
   * \brief Get the \ref coordinates::pattern associated with \ref indexible object T at a given index.
   * \details If the index exceeds count_indices(t), the result is
   * <code>coordinates::Dimensions<1>{}</code>, <code>coordinates::Any{1UZ}</code>, or <code>1UZ</code>.
   */
#ifdef __cpp_concepts
  template<indexible T, values::index I = std::integral_constant<std::size_t, 0>>
  constexpr coordinates::pattern decltype(auto)
#else
  template<typename T, typename I = std::integral_constant<std::size_t, 0>,
    std::enable_if_t<indexible<T> and values::index<I>, int> = 0>
  constexpr decltype(auto)
#endif
  get_index_pattern(T&& t, I i = {})
  {
    if constexpr (values::size_compares_with<I, index_count<T>, &stdex::is_gteq>)
    {
      return coordinates::Dimensions<1>{};
    }
    else if constexpr (interface::get_pattern_collection_defined_for<std::remove_reference_t<T>>)
    {
      decltype(auto) pat = get_pattern_collection(std::forward<T>(t));
      using Pat = decltype(pat);
      if constexpr (values::size_compares_with<I, collections::size_of<Pat>, &stdex::is_lt>)
        return collections::get_element(std::forward<Pat>(pat), i);
      else if constexpr (values::size_compares_with<I, collections::size_of<Pat>, &stdex::is_gteq>)
        return coordinates::Dimensions<1>{};
      else if (i < collections::get_size(pat))
        return coordinates::Any {collections::get_element(std::forward<Pat>(pat), i)};
      else
        return coordinates::Any {1_uz};
    }
    else if constexpr (values::size_compares_with<I, index_count<T>, &stdex::is_lt>)
    {
      constexpr auto ex = std::decay_t<decltype(get_mdspan(t))>::static_extent(i);
      if constexpr (ex == stdex::dynamic_extent)
        return get_mdspan(t).extent(static_cast<std::size_t>(i));
      else
        return coordinates::Dimensions<ex>{};
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
  template<std::size_t i, indexible T>
  constexpr coordinates::pattern auto
#else
  template<std::size_t i, typename T, std::enable_if_t<indexible<T>, int> = 0>
  constexpr auto
#endif
  get_index_pattern(T&& t)
  {
    return get_index_pattern(t, std::integral_constant<std::size_t, i>{});
  }

}

#endif
