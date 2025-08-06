/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition of \ref get_uniform_pattern_component
 */

#ifndef OPENKALMAN_GET_UNIFORM_PATTERN_COMPONENT_HPP
#define OPENKALMAN_GET_UNIFORM_PATTERN_COMPONENT_HPP

#include <optional>
#include <algorithm>
#include "collections/collections.hpp"
#include "linear-algebra/coordinates/concepts/pattern.hpp"
#include "linear-algebra/coordinates/concepts/fixed_pattern.hpp"
#include "linear-algebra/coordinates/concepts/descriptor_collection.hpp"
#include "linear-algebra/coordinates/functions/compare.hpp"
#include "linear-algebra/coordinates/traits/common_descriptor_type.hpp"

namespace OpenKalman::coordinates
{
  namespace detail
  {
    template<typename A, typename T, std::size_t i, std::size_t...is>
    constexpr auto
    equal_tuple_elements(const T& t, std::index_sequence<i, is...>)
    {
      auto t0 = OpenKalman::internal::generalized_std_get<i>(t);
      if ((... and stdcompat::is_eq(compare(t0, OpenKalman::internal::generalized_std_get<is>(t))))) return std::optional {A{t0}};
      return std::optional<A>{};
    }
  }


  /**
   * \brief If the argument is a uniform pattern, return the 1D component that can be replicated to produce the argument.
   * \details The result will equal the argument if replicated some number of times (including zero times).
   * \returns An std::optional object containing the 1D component, if it exists.
   */
#ifdef __cpp_concepts
  template<pattern T>
#else
  template<typename T, std::enable_if_t<pattern<T>, int> = 0>
#endif
  constexpr auto
  get_uniform_pattern_component(T&& t)
  {
    if constexpr (euclidean_pattern<T>)
    {
      return std::optional {Dimensions<1>{}};
    }
    else if constexpr (descriptor<T>)
    {
      if constexpr (dimension_of_v<T> == 1)
        return std::optional {std::forward<T>(t)};
      else
        return std::optional<std::decay_t<T>>{};
    }
    else // if constexpr (descriptor_collection<T>)
    {
      using C = common_descriptor_type_t<T>;
      using A = Any<typename internal::is_Any<C>::scalar_type>;
      if constexpr (dimension_of_v<C> == 1)
      {
        return std::optional {C{}};
      }
      else if constexpr (dimension_of_v<C> != dynamic_size or not collections::sized<T>)
      {
        return std::optional<A>{};
      }
      else if constexpr (collections::size_of_v<T> == 0)
      {
        return std::optional<C>{};
      }
      else if constexpr (collections::tuple_like<T>)
      {
        return detail::equal_tuple_elements<A>(t, std::make_index_sequence<std::tuple_size_v<T>>{});
      }
      else
      {
        if (get_is_euclidean(t)) return std::optional {A{Dimensions<1>{}}};
        decltype(auto) v = collections::views::all(std::forward<T>(t));
#ifdef __cpp_lib_ranges
        if (std::ranges::adjacent_find(v, std::ranges::not_equal_to{}) == v.end()) return std::optional {A{v.front()}};
#else
        if (std::adjacent_find(v.begin(), v.end(), std::not_equal_to{}) == v.end()) return std::optional {A{v.front()}};
#endif
        return std::optional<A>{};
      }
    }
  }

}

#endif
