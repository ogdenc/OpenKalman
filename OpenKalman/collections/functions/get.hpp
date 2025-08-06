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

#include "values/values.hpp"
#include "collections/concepts/collection.hpp"
#include "collections/traits/size_of.hpp"

namespace OpenKalman::collections
{
#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename I>
    inline constexpr bool gettable_with_i = []
    {
      if constexpr (values::fixed<I>) { return gettable<values::fixed_number_of_v<I>, T>; }
      else { return false; }
    }();
  }
#endif


  /**
   * \brief A generalization of std::get
   * \details This function takes a \ref values::value parameter instead of a template parameter like std::get.
   * - If the argument has a <code>get()</code> member, call that member.
   * - Otherwise, call <code>get&lt;i*gt;(std::forward&lt;Arg&gt;(arg))</code> if such a function is found using ADL.
   * - Otherwise, call <code>std::get&lt;i*gt;(std::forward&lt;Arg&gt;(arg))</code> if it is defined.
   * - Otherwise, call <code>std::ranges::begin(std::forward&lt;Arg&gt;(arg))</code> if it is a valid call.
   * \note This function performs no runtime bounds checking.
   */
#ifdef __cpp_concepts
  template<collection Arg, values::index I> requires
    (stdcompat::ranges::random_access_range<Arg> or (values::fixed<I> and gettable<values::fixed_number_of_v<I>, Arg>))
#else
  template<typename Arg, typename I, std::enable_if_t<collection<Arg> and values::index<I> and
    (stdcompat::ranges::random_access_range<Arg> or detail::gettable_with_i<Arg, I>), int> = 0>
#endif
  constexpr decltype(auto)
  get(Arg&& arg, I i)
  {
    if constexpr (sized<Arg> and values::fixed<I>)
    { static_assert(size_of_v<Arg> == dynamic_size or values::fixed_number_of_v<I> < size_of_v<Arg>, "Index out of range"); }

#ifdef __cpp_lib_ranges
    if constexpr (values::fixed<I> and requires { requires gettable<values::fixed_number_of<I>::value, Arg>; })
#else
    if constexpr (detail::gettable_with_i<Arg, I>)
#endif
    {
      return OpenKalman::internal::generalized_std_get<values::fixed_number_of_v<I>>(std::forward<Arg>(arg));
    }
    else
    {
      std::size_t n = values::to_number(std::move(i));
      if constexpr (std::is_array_v<stdcompat::remove_cvref_t<Arg>>)
      {
        return std::forward<Arg>(arg)[n];
      }
      else if constexpr (stdcompat::ranges::borrowed_range<Arg>)
      {
        return stdcompat::ranges::begin(std::forward<Arg>(arg))[n];
      }
      else
      {
        using namespace std;
        return begin(std::forward<Arg>(arg))[n];
      }
    }
  }


}

#endif
