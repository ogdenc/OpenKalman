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
 * \brief Definition of \ref count_indices.
 */

#ifndef OPENKALMAN_COUNT_INDICES_HPP
#define OPENKALMAN_COUNT_INDICES_HPP

#include "values/values.hpp"
#include "linear-algebra/concepts/indexible.hpp"
#include "linear-algebra/traits/get_mdspan.hpp"

namespace OpenKalman
{
  namespace detail
  {
    template<typename Mdspan, std::size_t i = Mdspan::rank()>
    constexpr auto remove_trailing_1D_indices()
    {
      if constexpr (i == 0)
        return std::integral_constant<std::size_t, i> {};
      else if constexpr (Mdspan::static_extent(i - 1_uz) == 1_uz)
        return remove_trailing_1D_indices<Mdspan, i - 1_uz>();
      else
        return std::integral_constant<std::size_t, i> {};
    }
  }


  /**
   * \brief Get the number of indices necessary to address all the components of an \ref indexible object.
   * \sa index_count
   */
#ifdef __cpp_concepts
  template<indexible T>
  constexpr values::index auto
#else
  template<typename T, std::enable_if_t<indexible<T>, int> = 0>
  constexpr auto
#endif
  count_indices(const T&)
  {
    using Mdspan = std::decay_t<decltype(get_mdspan(std::declval<const T&>()))>;
    return detail::remove_trailing_1D_indices<Mdspan>();
  }

}

#endif
