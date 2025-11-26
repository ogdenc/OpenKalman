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
 * \brief Definition of \ref ranges::reverse_view and \ref ranges::views::reverse.
 */

#ifndef OPENKALMAN_COMPATIBILITY_VIEWS_REVERSE_HPP
#define OPENKALMAN_COMPATIBILITY_VIEWS_REVERSE_HPP

#include <type_traits>
#include "view-concepts.hpp"
#include "view_interface.hpp"
#include "range_adaptor_closure.hpp"
#include "all.hpp"

namespace OpenKalman::stdex::ranges
{
#ifdef __cpp_lib_ranges
  using std::ranges::reverse_view;
  namespace views
  {
    using std::ranges::views::reverse;
  }
#else
  /**
   * \brief Equivalent to std::ranges::reverse_view.
   */
  template<typename V>
  struct reverse_view : stdex::ranges::view_interface<reverse_view<V>>
  {
    template<bool Enable = true, std::enable_if_t<Enable and stdex::default_initializable<V>, int> = 0>
    constexpr reverse_view() {}

    constexpr reverse_view(V r) : base_ {std::move(r)} {}


    template<bool Enable = true, std::enable_if_t<Enable and stdex::copy_constructible<V>, int> = 0>
    constexpr V base() const& { return base_; }

    constexpr V base() && { return std::move(base_); }


    template<bool Enable = true, std::enable_if_t<Enable and not stdex::ranges::common_range<V>, int> = 0>
    constexpr std::reverse_iterator<stdex::ranges::iterator_t<V>>
    begin()
    { return std::make_reverse_iterator(stdex::ranges::next(stdex::ranges::begin(base_), stdex::ranges::end(base_))); }

    template<bool Enable = true, std::enable_if_t<Enable and stdex::ranges::common_range<V>, int> = 0>
    constexpr std::reverse_iterator<stdex::ranges::iterator_t<V>>
    begin() { return std::make_reverse_iterator(stdex::ranges::end(base_)); }

    template<bool Enable = true, std::enable_if_t<Enable and stdex::ranges::common_range<const V>, int> = 0>
    constexpr auto
    begin() const { return std::make_reverse_iterator(stdex::ranges::end(base_)); }


    constexpr std::reverse_iterator<stdex::ranges::iterator_t<V>>
    end() { return std::make_reverse_iterator(stdex::ranges::begin(base_)); }

    template<bool Enable = true, std::enable_if_t<Enable and stdex::ranges::common_range<const V>, int> = 0>
    constexpr auto
    end() const { return std::make_reverse_iterator(stdex::ranges::begin(base_)); }


    template<bool Enable = true, std::enable_if_t<Enable and stdex::ranges::sized_range<const V>, int> = 0>
    constexpr auto
    size() { return stdex::ranges::size(base_); }

    template<bool Enable = true, std::enable_if_t<Enable and stdex::ranges::sized_range<const V>, int> = 0>
    constexpr auto
    size() const { return stdex::ranges::size(base_); }

  private:

    V base_;
  };


  /**
   * \brief Deduction guide
   */
  template<typename R>
  reverse_view(R&&) -> reverse_view<views::all_t<R>>;


  template<typename T>
  constexpr bool enable_borrowed_range<reverse_view<T>> = enable_borrowed_range<T>;


  namespace views
  {
    namespace detail
    {
      struct reverse_closure : stdex::ranges::range_adaptor_closure<reverse_closure>
      {
        template<typename R, std::enable_if_t<viewable_range<R>, int> = 0>
        constexpr auto
        operator() [[nodiscard]] (R&& r) const
        {
          return reverse_view {all(std::forward<R>(r))};
        }


        template<typename T>
        constexpr auto
        operator() [[nodiscard]] (const reverse_view<T>& r) const
        {
          return r.base();
        }


        template<typename T>
        constexpr auto
        operator() [[nodiscard]] (reverse_view<T>&& r) const
        {
          return std::move(r).base();
        }
      };

    }


    /**
     * \brief a RangeAdapterObject associated with \ref reverse_view.
     * \details The expression <code>views::reverse(arg)</code> is expression-equivalent
     * to <code>reverse_view(arg)</code> for any suitable \ref collection arg, except that it unwraps
     * reversed views if possible.
     * \sa reverse_view
     */
    inline constexpr detail::reverse_closure reverse;

  }

#endif
}



#endif
