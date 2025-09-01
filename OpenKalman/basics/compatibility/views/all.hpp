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
 * \internal
 * \brief Definition of \ref stdcompat::ranges::views::all and \ref stdcompat::ranges::views::all_t.
 */

#ifndef OPENKALMAN_COMPATIBILITY_VIEWS_ALL_HPP
#define OPENKALMAN_COMPATIBILITY_VIEWS_ALL_HPP

#include <type_traits>
#include "basics/compatibility/language-features.hpp"
#include "view-concepts.hpp"
#include "range_adaptor_closure.hpp"
#include "ref_view.hpp"
#include "owning_view.hpp"

namespace OpenKalman::stdcompat::ranges::views
{
#ifdef __cpp_lib_ranges
  using std::ranges::views::all;
  using std::ranges::views::all_t;
#else
  namespace detail
  {
    template<typename R, typename = void, typename = void>
    struct can_ref_view : std::false_type {};

    template<typename R>
    struct can_ref_view<R, std::enable_if_t<std::is_object_v<stdcompat::remove_cvref_t<R>> and range<stdcompat::remove_cvref_t<R>>>,
      std::void_t<decltype(ref_view {std::declval<R>()})>> : std::true_type {};


    struct all_closure : range_adaptor_closure<all_closure>
    {
      template<typename R, std::enable_if_t<viewable_range<R>, int> = 0>
      constexpr auto
      operator() [[nodiscard]] (R&& r) const
      {
        if constexpr (view<std::decay_t<R>>)
          return static_cast<std::decay_t<R>>(std::forward<R>(r));
        else if constexpr (can_ref_view<R>::value)
          return ref_view {std::forward<R>(r)};
        else
          return owning_view {std::forward<R>(r)};
      }
    };
  }


  /**
   * \brief Equivalent to std::ranges::views::all.
   * \internal
   */
  inline constexpr detail::all_closure all;


  /**
   * \brief Equivalent to std::ranges::views::all_t.
   * \internal
   */
  template<typename R, std::enable_if_t<viewable_range<R>, int> = 0>
  using all_t = decltype(all(std::declval<R>()));

#endif
}

#endif
