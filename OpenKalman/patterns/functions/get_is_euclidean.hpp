/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2026 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref get_is_euclidean.
 */

#ifndef OPENKALMAN_GET_IS_EUCLIDEAN_HPP
#define OPENKALMAN_GET_IS_EUCLIDEAN_HPP

#include "collections/collections.hpp"
#include "patterns/concepts/pattern.hpp"
#include "patterns/concepts/descriptor.hpp"
#include "patterns/functions/internal/get_descriptor_is_euclidean.hpp"

namespace OpenKalman::patterns
{
  namespace detail
  {
#ifndef __cpp_lib_ranges
    template<typename T, typename = void>
    struct is_unbound_fixed_range : std::false_type {};

    template<typename T>
    struct is_unbound_fixed_range<T, std::enable_if_t<stdex::ranges::range<T>>>
      : std::bool_constant<values::fixed<decltype(patterns::internal::get_descriptor_is_euclidean(
        std::declval<stdex::ranges::range_value_t<T>>()))>> {};
#endif


    template<std::size_t i = 0, typename T>
    static constexpr auto get_is_euclidean_fixed(const T& t)
    {
      if constexpr (i < collections::size_of_v<T>)
      {
        return values::operation(
          std::logical_and{},
          internal::get_descriptor_is_euclidean(collections::get<i>(t)),
          get_is_euclidean_fixed<i + 1>(t));
      }
      else return std::true_type {};
    }
  }


  /**
   * \brief Determine, whether \ref patterns::pattern Arg is euclidean.
   */
#ifdef __cpp_concepts
  template<pattern Arg> requires descriptor<Arg> or collections::sized<Arg> or
    values::fixed<decltype(internal::get_descriptor_is_euclidean(std::declval<stdex::ranges::range_value_t<Arg>>()))>
#else
  template<typename Arg, std::enable_if_t<pattern<Arg> and
    (descriptor<Arg> or collections::sized<Arg> or detail::is_unbound_fixed_range<Arg>::value), int> = 0>
#endif
  constexpr auto
  get_is_euclidean(const Arg& arg)
  {
    if constexpr (descriptor<Arg>)
    {
      return internal::get_descriptor_is_euclidean(arg);
    }
    else if constexpr (values::fixed_value_compares_with<collections::size_of<Arg>, 0>)
    {
      return std::true_type {};
    }
    else if constexpr (not collections::sized<Arg> or values::fixed_value_compares_with<collections::size_of<Arg>, stdex::dynamic_extent>)
    {
      using C = decltype(internal::get_descriptor_is_euclidean(std::declval<stdex::ranges::range_value_t<Arg>>()));
      if constexpr (values::fixed<C>)
        return values::fixed_value_of<C>{};
      else
#ifdef __cpp_lib_ranges_fold
        return std::ranges::fold_left(collections::views::all(arg), true,
          [](const auto& a, const auto& b) { return a and internal::get_descriptor_is_euclidean(b); });
#else
      {
        for (const auto& c : arg) if (not internal::get_descriptor_is_euclidean(c)) return false;
        return true;
      }
#endif
    }
    else
    {
      return detail::get_is_euclidean_fixed(arg);
    }
  }


}

#endif
