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
 * \brief Definition of \ref basics::compatibility::ranges::ref_view.
 */

#ifndef OPENKALMAN_COMPATIBILITY_VIEWS_REF_VIEW_HPP
#define OPENKALMAN_COMPATIBILITY_VIEWS_REF_VIEW_HPP

#ifndef __cpp_lib_ranges

#include "basics/compatibility/ranges.hpp"
#include "view_interface.hpp"

namespace OpenKalman::ranges
{
  /**
   * \internal
   * \brief Equivalent to std::ranges::ref_view.
   */
  template<typename R>
  struct ref_view : view_interface<ref_view<R>>
  {
  private:

    static_assert(std::is_object_v<R>);
    static void FUN(R&);
    static void FUN(R&&) = delete;

  public:

    /**
     * \brief Construct from a \ref range.
     */
    template<typename T, std::enable_if_t<std::is_convertible_v<T, R&> and (not std::is_same_v<remove_cvref_t<T>, ref_view>), int> = 0,
      typename = std::void_t<decltype(FUN(std::declval<T>()))>>
    constexpr
    ref_view(T&& t) : r_ {std::addressof(static_cast<R&>(std::forward<T>(t)))} {}


    /**
     * \returns A reference to the wrapped object.
     */
    constexpr R&
    base() const { return *r_; }


    /**
     * \returns An iterator at the beginning, if the base object is a range.
     */
    constexpr ranges::iterator_t<R> begin() const { return ranges::begin(*r_); }


    /**
     * \returns An iterator at the end, if the base object is a range.
     */
    constexpr ranges::sentinel_t<R> end() const { return ranges::end(*r_); }


    /**
     * \brief Indicates whether the view is empty
     */
    constexpr bool empty() const { return ranges::empty(*r_); }


    /**
     * \brief The size of the object.
     */
    template<bool Enable = true, std::enable_if_t<Enable and sized_range<R>, int> = 0>
    constexpr auto size() const { return ranges::size(*r_); }

  private:

    R* r_;

  };


  /**
   * \brief Deduction guide.
   */
  template<typename R>
  ref_view(R&) -> ref_view<R>;


  template<typename R>
  constexpr bool enable_borrowed_range<ref_view<R>> = true;

}


#endif
#endif //OPENKALMAN_COMPATIBILITY_VIEWS_REF_VIEW_HPP
