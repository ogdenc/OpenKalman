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
 * \brief Definition for \ref collections::get.
 */

#ifndef OPENKALMAN_COLLECTIONS_GET_HPP
#define OPENKALMAN_COLLECTIONS_GET_HPP

#include <tuple>
#include "basics/global-definitions.hpp"
#include "basics/ranges.hpp"
#include "values/concepts/fixed.hpp"
#include "values/concepts/index.hpp"
#include "values/traits/fixed_number_of.hpp"
#include "collections/concepts/collection.hpp"
#include "collections/traits/size_of.hpp"

namespace OpenKalman::collections
{
  /**
   * \brief A generalization of std::get
   * \details This function takes a \ref value::value parameter instead of a template parameter like std::get.
   * - If the argument has a <code>get()</code> member, call that member.
   * - Otherwise, call <code>get&lt;i*gt;(std::forward&lt;Arg&gt;(arg))</code> if such a function is found using ADL.
   * - Otherwise, call <code>std::get&lt;i*gt;(std::forward&lt;Arg&gt;(arg))</code> if it is defined.
   * - Otherwise, call <code>std::ranges::begin(std::forward&lt;Arg&gt;(arg))</code> if it is a valid call.
   */
#ifdef __cpp_lib_ranges
  template<collection Arg, value::index I> requires value::fixed<I> or sized_random_access_range<Arg>
#else
  template<typename Arg, typename I, std::enable_if_t<collection<Arg> and value::index<I> and
    (value::fixed<I> or sized_random_access_range<Arg>), int> = 0>
#endif
  constexpr decltype(auto)
  get(Arg&& arg, I i)
  {
    if constexpr (value::fixed<I> and size_of_v<Arg> != dynamic_size)
      static_assert(value::fixed_number_of_v<I> < size_of_v<Arg>, "Index out of range");

    if constexpr (tuple_like<Arg> and value::fixed<I>)
    {
      return OpenKalman::internal::generalized_std_get<value::fixed_number_of_v<I>>(std::forward<Arg>(arg));
    }
    else
    {
#ifdef __cpp_lib_ranges
      using namespace std;
#endif
      std::size_t n = value::to_number(std::move(i));
      if constexpr (std::is_array_v<remove_cvref_t<Arg>>)
        return std::forward<Arg>(arg)[n];
      else if constexpr (ranges::borrowed_range<Arg>)
        return ranges::begin(std::forward<Arg>(arg))[n];
      else
        return begin(std::forward<Arg>(arg))[n];
    }
  }


} // namespace OpenKalman::collections

#endif //OPENKALMAN_COLLECTIONS_GET_HPP
