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
 * \brief Definition for \ref get_is_euclidean.
 */

#ifndef OPENKALMAN_GET_IS_EUCLIDEAN_HPP
#define OPENKALMAN_GET_IS_EUCLIDEAN_HPP

#include "collections/collections.hpp"
#include "linear-algebra/coordinates/interfaces/coordinate_descriptor_traits.hpp"
#include "linear-algebra/coordinates/concepts/pattern.hpp"
#include "linear-algebra/coordinates/concepts/descriptor.hpp"
#include "linear-algebra/coordinates/functions/internal/get_descriptor_is_euclidean.hpp"

namespace OpenKalman::coordinates
{
  namespace detail
  {
    template<std::size_t i = 0, typename Tup>
    static constexpr auto get_is_euclidean_tuple(const Tup& tup)
    {
      if constexpr (i < std::tuple_size_v<Tup>)
      {
        return values::operation(
          std::logical_and{},
          coordinates::internal::get_descriptor_is_euclidean(OpenKalman::internal::generalized_std_get<i>(tup)),
          get_is_euclidean_tuple<i + 1>(tup));
      }
      else return std::true_type {};
    }
  } // namespace detail


  /**
   * \brief Determine, whether \ref coordinates::pattern Arg is euclidean.
   */
#ifdef __cpp_concepts
  template<pattern Arg>
#else
  template<typename Arg, std::enable_if_t<pattern<Arg>, int> = 0>
#endif
  constexpr auto
  get_is_euclidean(const Arg& arg)
  {
    if constexpr (descriptor<Arg>)
    {
      return coordinates::internal::get_descriptor_is_euclidean(arg);
    }
    else if constexpr (values::fixed_number_compares_with<collections::size_of<Arg>, 0>)
    {
      return std::true_type {};
    }
    else if constexpr (not collections::sized<Arg> and stdcompat::ranges::range<Arg>)
    {
      using V = stdcompat::ranges::range_value_t<stdcompat::remove_cvref_t<Arg>>;
      if constexpr (stdcompat::default_initializable<V>)
        return coordinates::internal::get_descriptor_is_euclidean(V{});
      else
        return std::false_type {};
    }
    else if constexpr (collections::tuple_like<Arg>)
    {
      return detail::get_is_euclidean_tuple(arg);
    }
    else
    {
#ifdef __cpp_lib_ranges_fold
      return std::ranges::fold_left(collections::views::all(arg), true,
        [](const auto& a, const auto& b) { return a and coordinates::internal::get_descriptor_is_euclidean(b); });
#else
      for (const auto& c : arg) if (not coordinates::internal::get_descriptor_is_euclidean(c)) return false;
      return true;
#endif
    }
  }


}


#endif
