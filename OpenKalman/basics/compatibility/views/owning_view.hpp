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
 * \brief Definition of \ref collections::owning_view.
 */

#ifndef OPENKALMAN_COMPATIBILITY_VIEWS_OWNING_VIEW_HPP
#define OPENKALMAN_COMPATIBILITY_VIEWS_OWNING_VIEW_HPP

#include "basics/compatibility/internal/exposition.hpp"
#include "view_interface.hpp"

namespace OpenKalman::stdex::ranges
{
#ifdef __cpp_lib_ranges
  using std::ranges::owning_view;
#else
  /**
   * \internal
   * \brief Equivalent to std::ranges::owning_view.
   */
  template<typename R>
  struct owning_view : view_interface<owning_view<R>>
  {
    static_assert(std::is_object_v<std::remove_reference_t<R>> and std::is_move_constructible_v<std::remove_reference_t<R>> and
      std::is_assignable_v<std::remove_reference_t<R>&, std::remove_reference_t<R>> and
      not OpenKalman::internal::is_initializer_list<R>::value);


    /**
     * \brief Default constructor.
     */
    template<bool Enable = true, std::enable_if_t<Enable and stdex::default_initializable<R>, int> = 0>
    constexpr
    owning_view() {}


    /**
     * \brief Move constructor.
     */
    constexpr
    owning_view(owning_view&& other) = default;


    /**
     * \brief Construct from a \ref collection.
     */
    explicit constexpr
    owning_view(R&& t) : my_r {std::forward<R>(t)} {}


    /**
     * \brief Move assignment operator.
     */
    constexpr owning_view&
    operator=(owning_view&& other) = default;


    /**
     * \brief Get the base object.
     */
    constexpr decltype(auto)
    base() & { return my_r; }

    /// \overload
    constexpr decltype(auto)
    base() const & { return my_r; }

    /// \overload
    constexpr decltype(auto)
    base() && noexcept { return std::move(*this).my_r; }

    /// \overload
    constexpr decltype(auto)
    base() const && noexcept { return std::move(*this).my_r; }


    /**
     * \returns An iterator at the beginning, if the base object is a range.
     */
    constexpr auto begin() { return stdex::ranges::begin(my_r); }

    /// \overload
    template<bool Enable = true, std::enable_if_t<Enable and range<const R>, int> = 0>
    constexpr auto begin() const { return stdex::ranges::begin(my_r); }


    /**
     * \returns An iterator at the end, if the base object is a range.
     */
    constexpr auto end() { return stdex::ranges::end(my_r); }

    /// \overload
    template<bool Enable = true, std::enable_if_t<Enable and range<const R>, int> = 0>
    constexpr auto end() const { return stdex::ranges::end(my_r); }


    /**
     * \brief Indicates whether the view is empty
     */
    template<typename Enable = void, typename = std::void_t<Enable, decltype(stdex::ranges::empty(std::declval<R>()))>>
    constexpr auto empty() { return stdex::ranges::empty(my_r); }

    /// \overload
    template<typename Enable = void, typename = std::void_t<Enable, decltype(stdex::ranges::empty(std::declval<const R>()))>>
    constexpr auto empty() const { return stdex::ranges::empty(my_r); }


    /**
     * \brief The size of the object.
     */
    template<bool Enable = true, std::enable_if_t<Enable and sized_range<R>, int> = 0>
    constexpr auto size() noexcept { return stdex::ranges::size(my_r); }

    /// \overload
    template<bool Enable = true, std::enable_if_t<Enable and sized_range<const R>, int> = 0>
    constexpr auto size() const noexcept { return stdex::ranges::size(my_r); }

  private:

   R my_r;

  };


  template<typename R>
  constexpr bool enable_borrowed_range<owning_view<R>> = enable_borrowed_range<R>;

#endif
}

#endif
