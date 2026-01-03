/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition of \ref is_one_dimensional function.
 */

#ifndef OPENKALMAN_IS_ONE_DIMENSIONAL_HPP
#define OPENKALMAN_IS_ONE_DIMENSIONAL_HPP

#include "linear-algebra/traits/count_indices.hpp"
#include "linear-algebra/traits/index_count.hpp"
#include "linear-algebra/traits/get_index_extent.hpp"

namespace OpenKalman
{
  namespace detail
  {
    template<std::size_t i = 0, typename T>
    constexpr bool is_one_dimensional_impl(const T& t)
    {
      if constexpr (i < index_count_v<T>)
      {
        return values::operation(
          std::logical_and{},
          values::operation(std::equal_to{}, get_index_extent<i>(t), std::integral_constant<std::size_t, 1>{}),
          is_one_dimensional_impl<i + 1>(t));
      }
      else
      {
        return std::true_type {};
      }
    }
  }


  /**
   * \brief Determine whether T is one_dimensional, meaning that every index has a dimension of 1.
   * \details Each index need not have an equivalent \ref patterns::pattern.
   */
#ifdef __cpp_concepts
  template<indexible T>
  constexpr internal::boolean_testable auto is_one_dimensional(const T& t)
#else
  template<typename T, std::enable_if_t<indexible<T>, int> = 0>
  constexpr auto is_one_dimensional(const T& t)
#endif
  {
    if constexpr (index_count_v<T> == stdex::dynamic_extent)
    {
      for (std::size_t i = 1; i < count_indices(t); ++i) if (get_index_extent(t, i) != 1) return false;
      return true;
    }
    else
    {
      return detail::is_one_dimensional_impl(t);
    }

  }

}

#endif
