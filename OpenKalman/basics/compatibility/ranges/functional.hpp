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
 * \brief Definition of callable objects equivalent to std::ranges::equal_to, etc.
 */

#ifndef OPENKALMAN_COMPATIBILITY_RANGES_FUNCTIONAL_HPP
#define OPENKALMAN_COMPATIBILITY_RANGES_FUNCTIONAL_HPP

#include "basics/compatibility/comparison.hpp"

namespace OpenKalman::stdcompat::ranges
{
#ifdef __cpp_lib_ranges
  using std::ranges::equal_to;
  using std::ranges::not_equal_to;
  using std::ranges::greater;
  using std::ranges::less;
  using std::ranges::greater_equal;
  using std::ranges::less_equal;
#else
  struct equal_to
  {
    template<typename Tp, typename Up, std::enable_if_t<stdcompat::equality_comparable_with<Tp, Up>, int> = 0>
    constexpr bool
    operator()(Tp&& t, Up&& u) const noexcept(noexcept(std::declval<Tp>() == std::declval<Up>()))
    {
      return std::forward<Tp>(t) == std::forward<Up>(u);
    }

    struct is_transparent;
  };


  struct not_equal_to
  {
    template<typename Tp, typename Up, std::enable_if_t<stdcompat::equality_comparable_with<Tp, Up>, int> = 0>
    constexpr bool
    operator()(Tp&& t, Up&& u) const noexcept(noexcept(std::declval<Up>() == std::declval<Tp>()))
    {
      return not equal_to{}(std::forward<Tp>(t), std::forward<Up>(u));
    }

    struct is_transparent;
  };


  struct less
  {
  private:

    template<typename T, typename U, typename = void>
    struct has_deduced_lt : std::false_type {};

    template<typename T, typename U>
    struct has_deduced_lt<T, U, std::void_t<decltype(operator<(std::declval<T>(), std::declval<U>()))>> : std::true_type {};

    template<typename T, typename U, typename = void>
    struct has_member_lt : std::false_type {};

    template<typename T, typename U>
    struct has_member_lt<T, U, std::void_t<decltype(std::declval<T>().operator<(std::declval<U>()))>> : std::true_type {};

  public:

    template<typename Tp, typename Up, std::enable_if_t<stdcompat::totally_ordered_with<Tp, Up>, int> = 0>
    constexpr bool
    operator()(Tp&& t, Up&& u) const noexcept(noexcept(std::declval<Tp>() < std::declval<Up>()))
    {
      return std::forward<Tp>(t) < std::forward<Up>(u);
    }

    struct is_transparent;
  };


  struct greater
  {
    template<typename Tp, typename Up, std::enable_if_t<stdcompat::totally_ordered_with<Tp, Up>, int> = 0>
    constexpr bool
    operator()(Tp&& t, Up&& u) const noexcept(noexcept(std::declval<Up>() < std::declval<Tp>()))
    {
      return less{}(std::forward<Up>(u), std::forward<Tp>(t));
    }

    struct is_transparent;
  };


  struct greater_equal
  {
    template<typename Tp, typename Up, std::enable_if_t<stdcompat::totally_ordered_with<Tp, Up>, int> = 0>
    constexpr bool
    operator()(Tp&& t, Up&& u) const noexcept(noexcept(std::declval<Tp>() < std::declval<Up>()))
    {
      return not less{}(std::forward<Tp>(t), std::forward<Up>(u));
    }

    struct is_transparent;
  };


  struct less_equal
  {
    template<typename Tp, typename Up, std::enable_if_t<stdcompat::totally_ordered_with<Tp, Up>, int> = 0>
    constexpr bool
    operator()(Tp&& t, Up&& u) const noexcept(noexcept(std::declval<Up>() < std::declval<Tp>()))
    {
      return not less{}(std::forward<Up>(u), std::forward<Tp>(t));
    }

    struct is_transparent;
  };

#endif

}

#endif
