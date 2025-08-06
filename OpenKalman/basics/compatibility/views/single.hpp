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
 * \brief Definition of \ref ranges::single_view and \ref ranges::views::single.
 */

#ifndef OPENKALMAN_COMPATIBILITY_VIEWS_SINGLE_HPP
#define OPENKALMAN_COMPATIBILITY_VIEWS_SINGLE_HPP

#include "view-concepts.hpp"
#include "view_interface.hpp"

namespace OpenKalman::stdcompat::ranges
{
#ifdef __cpp_lib_ranges
  using std::ranges::single_view;
  namespace views
  {
    using std::ranges::views::single;
  }
#else
  /**
   * \brief Equivalent to std::ranges::single_view.
   * \internal
   */
#ifdef __cpp_concepts
  template<std::move_constructible T> requires std::is_object_v<T>
#else
  template<typename T>
#endif
  struct single_view : view_interface<single_view<T>>
  {
    /**
     * \brief Default constructor
     */
#ifdef __cpp_concepts
    constexpr
    single_view() requires std::default_initializable<T> = default;
#else
    template<bool Enable = true, std::enable_if_t<Enable and stdcompat::default_initializable<T>, int> = 0>
    constexpr
    single_view() {}
#endif


    /**
     * \brief Construct from an object convertible to type T.
     */
#ifdef __cpp_concepts
    explicit constexpr
    single_view(const T& t) requires std::copy_constructible<T> : value_ {t} {}
    requires std::copy_constructible<T>
#else
    template<bool Enable = true, std::enable_if_t<stdcompat::copy_constructible<T>, int> = 0>
    explicit constexpr
    single_view(const T& t) : value_ {t} {}
#endif


    /**
     * \brief Construct from an object convertible to type T.
     */
    explicit constexpr
    single_view(T&& t) : value_ {std::move(t)} {}


    /**
     * \brief Equivalent to <code>data()</code>;
     */
    constexpr T*
    begin() noexcept { return data(); }

    /// \overload
    constexpr const T*
    begin() const noexcept { return data(); }


    /**
     * \brief Equivalent to <code>data() + 1</code>;
     */
    constexpr T*
    end() noexcept { return data() + 1_uz; }

    /// \overload
    constexpr const T*
    end() const noexcept { return data() + 1_uz; }


    /**
     * \returns <code>false</code>
     */
    static constexpr auto
    empty() noexcept { return false; }


    /**
     * \brief The size of the resulting object (which is always 1)
     */
    static constexpr std::size_t
    size() noexcept { return 1_uz; }


    /**
     * \brief A pointer to the contained value
     */
    constexpr T*
    data() noexcept { return std::addressof(*value_); }


    /**
     * \overload
     */
    constexpr const T*
    data() const noexcept { return std::addressof(*value_); }

  private:

    OpenKalman::internal::movable_box<T> value_;

  };


  /**
   * \brief Deduction guide
   */
  template<typename Arg>
  single_view(Arg) -> single_view<Arg>;


#ifdef __cpp_impl_three_way_comparison
  template<typename T>
  constexpr std::partial_ordering
  operator<=>(const single_view<T>& lhs, const T& rhs) noexcept
  {
    return get(lhs, std::integral_constant<std::size_t, 0>{}) <=> rhs;
  }

  template<typename T>
  constexpr bool
  operator==(const single_view<T>& lhs, const T& rhs) noexcept
  {
    return stdcompat::is_eq(operator<=>(lhs, rhs));
  }
#else
  template<typename T>
  constexpr bool operator==(const single_view<T>& lhs, const T& rhs) noexcept { return get(lhs, std::integral_constant<std::size_t, 0>{}) == rhs; }

  template<typename T>
  constexpr bool operator!=(const single_view<T>& lhs, const T& rhs) noexcept { return get(lhs, std::integral_constant<std::size_t, 0>{}) != rhs; }

  template<typename T>
  constexpr bool operator<(const single_view<T>& lhs, const T& rhs) noexcept { return get(lhs, std::integral_constant<std::size_t, 0>{}) < rhs; }

  template<typename T>
  constexpr bool operator>(const single_view<T>& lhs, const T& rhs) noexcept { return get(lhs, std::integral_constant<std::size_t, 0>{}) > rhs; }

  template<typename T>
  constexpr bool operator<=(const single_view<T>& lhs, const T& rhs) noexcept { return get(lhs, std::integral_constant<std::size_t, 0>{}) <= rhs; }

  template<typename T>
  constexpr bool operator>=(const single_view<T>& lhs, const T& rhs) noexcept { return get(lhs, std::integral_constant<std::size_t, 0>{}) >= rhs; }


  template<typename T>
  constexpr bool operator==(const T& lhs, const single_view<T>& rhs) noexcept { return lhs == get(rhs, std::integral_constant<std::size_t, 0>{}); }

  template<typename T>
  constexpr bool operator!=(const T& lhs, const single_view<T>& rhs) noexcept { return lhs != get(rhs, std::integral_constant<std::size_t, 0>{}); }

  template<typename T>
  constexpr bool operator<(const T& lhs, const single_view<T>& rhs) noexcept { return lhs < get(rhs, std::integral_constant<std::size_t, 0>{}); }

  template<typename T>
  constexpr bool operator>(const T& lhs, const single_view<T>& rhs) noexcept { return lhs > get(rhs, std::integral_constant<std::size_t, 0>{}); }

  template<typename T>
  constexpr bool operator<=(const T& lhs, const single_view<T>& rhs) noexcept { return lhs <= get(rhs, std::integral_constant<std::size_t, 0>{}); }

  template<typename T>
  constexpr bool operator>=(const T& lhs, const single_view<T>& rhs) noexcept { return lhs >= get(rhs, std::integral_constant<std::size_t, 0>{}); }

#endif


  namespace views
  {
    namespace detail
    {
      struct single_impl
      {
        template<typename R>
        constexpr auto
        operator() [[nodiscard]] (R&& r) const { return single_view<R> {std::forward<R>(r)}; }
      };
    }


    /**
     * \brief Equivalent to std::ranges::views::single.
     * \internal
     * \sa single_view
     */
    inline constexpr detail::single_impl single;

  }

#endif
}

#endif
