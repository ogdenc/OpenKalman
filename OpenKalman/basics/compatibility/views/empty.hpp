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
 * \brief Definition of \ref ranges::empty_view and \ref ranges::views::empty.
 */

#ifndef OPENKALMAN_COMPATIBILITY_VIEWS_EMPTY_HPP
#define OPENKALMAN_COMPATIBILITY_VIEWS_EMPTY_HPP

#ifndef __cpp_lib_ranges

#include "view-concepts.hpp"
#include "view_interface.hpp"

namespace OpenKalman::ranges
{
  /**
   * \brief Equivalent to std::ranges::single_view.
   * \internal
   */
  template<typename T>
  struct empty_view : ranges::view_interface<empty_view<T>>
  {
    static constexpr T* begin() noexcept { return nullptr; }

    static constexpr T* end() noexcept { return nullptr; }

    static constexpr T* data() noexcept { return nullptr; }

    static constexpr std::size_t size() noexcept { return 0; }

    static constexpr bool empty() noexcept { return true; }
  };


  template<typename T>
  constexpr bool enable_borrowed_range<OpenKalman::ranges::empty_view<T>> = true;

}


namespace OpenKalman::ranges::views
{
  /**
   * \brief Equivalent to std::ranges::views::empty.
   */
  template<class T>
  constexpr empty_view<T> empty{};

}


#endif

#endif //OPENKALMAN_COMPATIBILITY_VIEWS_EMPTY_HPP
