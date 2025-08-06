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
 * \brief Definition equivalent to std::ranges::view_interface
 */

#ifndef OPENKALMAN_COMPATIBILITY_RANGES_VIEW_INTERFACE_HPP
#define OPENKALMAN_COMPATIBILITY_RANGES_VIEW_INTERFACE_HPP

#include <type_traits>
#include "basics/compatibility/ranges.hpp"

namespace OpenKalman::stdcompat::ranges
{
#ifdef __cpp_lib_ranges
  using std::ranges::view_interface;
#else
  /**
   * \internal
   * \brief Equivalent to std::ranges::view_interface.
   */
  template<typename Derived>
  struct view_interface
  {
    template<typename D = Derived, std::enable_if_t<sized_range<D> or forward_range<D>, int> = 0>
    [[nodiscard]] constexpr bool
    empty()
    {
      auto& derived = static_cast<D&>(*this);
      if constexpr (sized_range<D>) return stdcompat::ranges::size(derived) == 0;
      else return begin(derived) == end(derived);
    }

    template<typename D = const Derived, std::enable_if_t<sized_range<D> or forward_range<D>, int> = 0>
    [[nodiscard]] constexpr bool
    empty() const
    {
      auto& derived = static_cast<D&>(*this);
      if constexpr (sized_range<D>) return stdcompat::ranges::size(derived) == 0;
      else return begin(derived) == end(derived);
    }


    template<typename D = Derived, std::enable_if_t<range<D>, int> = 0>
    constexpr auto
    cbegin() { return stdcompat::ranges::cbegin(static_cast<D&>(*this)); }

    template<typename D = const Derived, std::enable_if_t<range<D>, int> = 0>
    constexpr auto
    cbegin() const { return stdcompat::ranges::cbegin(static_cast<D&>(*this)); }


    template<typename D = Derived, std::enable_if_t<range<D>, int> = 0>
    constexpr auto
    cend() { return stdcompat::ranges::cend(static_cast<D&>(*this)); }

    template<typename D = const Derived, std::enable_if_t<range<D>, int> = 0>
    constexpr auto
    cend() const { return stdcompat::ranges::cend(static_cast<D&>(*this)); }


    template<typename D = Derived, typename = std::void_t<decltype(stdcompat::ranges::empty(std::declval<D&>()))>>
    constexpr explicit
    operator bool() { return not stdcompat::ranges::empty(static_cast<D&>(*this)); }

    template<typename D = const Derived, typename = std::void_t<decltype(stdcompat::ranges::empty(std::declval<D&>()))>>
    constexpr explicit
    operator bool() const { return not stdcompat::ranges::empty(static_cast<D&>(*this)); }


    template<typename D = Derived, std::enable_if_t<forward_range<D>, int> = 0,
      typename = std::void_t<decltype(end(std::declval<D&>()) - begin(std::declval<D&>()))>>
    constexpr auto
    size() { return end(static_cast<D&>(*this)) - begin(static_cast<D&>(*this)); }

    template<typename D = const Derived, std::enable_if_t<forward_range<D>, int> = 0,
      typename = std::void_t<decltype(end(std::declval<D&>()) - begin(std::declval<D&>()))>>
    constexpr auto
    size() const { return end(static_cast<D&>(*this)) - begin(static_cast<D&>(*this)); }


    template<typename D = Derived, std::enable_if_t<forward_range<D>, int> = 0>
    constexpr decltype(auto)
    front() { return *begin(static_cast<D&>(*this)); }

    template<typename D = const Derived, std::enable_if_t<forward_range<D>, int> = 0>
    constexpr decltype(auto)
    front() const { return *begin(static_cast<D&>(*this)); }


    template<typename D = Derived, std::enable_if_t<bidirectional_range<D> and common_range<D>, int> = 0>
    constexpr decltype(auto)
    back() { return *std::prev(end(static_cast<D&>(*this))); }

    template<typename D = const Derived, std::enable_if_t<bidirectional_range<D> and common_range<D>, int> = 0>
    constexpr decltype(auto)
    back() const { return *std::prev(end(static_cast<D&>(*this))); }


    template<typename D = Derived, std::enable_if_t<random_access_range<D>, int> = 0>
    constexpr decltype(auto)
    operator[](range_difference_t<D> n) { return begin(static_cast<D&>(*this))[n]; }

    template<typename D = const Derived, std::enable_if_t<random_access_range<D>, int> = 0>
    constexpr decltype(auto)
    operator[](range_difference_t<D> n) const { return begin(static_cast<D&>(*this))[n]; }

  };

#endif
}

#endif //OPENKALMAN_COMPATIBILITY_RANGES_VIEW_INTERFACE_HPP
