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
 * \brief Definition of \ref collections::single_view and \ref collections::views::single.
 */

#ifndef OPENKALMAN_COLLECTIONS_VIEWS_SINGLE_HPP
#define OPENKALMAN_COLLECTIONS_VIEWS_SINGLE_HPP

#include <type_traits>
#include <tuple>
#include "values/concepts/index.hpp"
#include "collection_view_interface.hpp"

namespace OpenKalman::collections
{
  /**
   * \internal
   * \brief A view that wraps an object and presents it as a one-element \ref collection.
   * \details If the base type is not itself a \ref collection, it will be presented as a singleton collection (only one component).
   * The following should compile:
   * \code
   * static_assert(single_view{4}[0] == 4);
   * static_assert(std::size_of_v<single_view<int>> == 1);
   * static_assert(equal_to{}(single_view{5}, std::vector{5}));
   * \endcode
   * \sa views::single
   */
  template<typename T>
  struct single_view : collection_view_interface<single_view<T>>
  {
#ifdef __cpp_concepts
    constexpr
    single_view() requires std::default_initializable<std::tuple<T>> = default;
#else
    template<typename aT = std::tuple<T>, std::enable_if_t<std::is_default_constructible_v<aT>, int> = 0>
    constexpr
    single_view() {}
#endif


    /**
     * \brief Construct from an object convertible to type T.
     */
#ifdef __cpp_concepts
    template<typename Arg> requires std::constructible_from<std::tuple<T>, Arg&&>
#else
    template<typename Arg, std::enable_if_t<std::is_constructible_v<std::tuple<T>, Arg&&>, int> = 0>
#endif
    explicit constexpr
    single_view(Arg&& arg) : my_t {std::forward<Arg>(arg)} {}


    /**
     * \brief Assign from an object convertible to type T.
     */
#ifdef __cpp_concepts
    template<typename Arg> requires std::constructible_from<std::tuple<T>, Arg&&> and std::is_move_assignable_v<std::tuple<T>>
#else
    template<typename Arg, std::enable_if_t<std::is_constructible_v<std::tuple<T>, Arg&&> and std::is_move_assignable<std::tuple<T>>::value, int> = 0>
#endif
    constexpr single_view&
    operator=(Arg&& arg) { my_t = std::tuple<T> {std::forward<Arg>(arg)}; return *this; }


    /**
     * \brief Get element i.
     */
#ifdef __cpp_explicit_this_parameter
    template<std::size_t i>
    constexpr decltype(auto)
    get(this auto&& self) noexcept
    {
      static_assert(i == 0, "Index out of range");
      return std::get<0>(std::forward<decltype(self)>(self).my_t);
    }
#else
    template<std::size_t i>
    constexpr decltype(auto)
    get() &
    {
      static_assert(i == 0, "Index out of range");
      return std::get<0>(my_t);
    }

    template<std::size_t i>
    constexpr decltype(auto)
    get() const &
    {
      static_assert(i == 0, "Index out of range");
      return std::get<0>(my_t);
    }

    template<std::size_t i>
    constexpr decltype(auto)
    get() && noexcept
    {
      static_assert(i == 0, "Index out of range");
      return std::get<0>(std::move(*this).my_t);
    }

    template<std::size_t i>
    constexpr decltype(auto)
    get() const && noexcept
    {
      static_assert(i == 0, "Index out of range");
      return std::get<0>(std::move(*this).my_t);
    }
#endif


#ifdef __cpp_concepts
    constexpr value::index auto size() const noexcept
#else
    constexpr auto size() const noexcept
#endif
    {
      return std::integral_constant<std::size_t, 1> {};
    }

  private:

    std::tuple<T> my_t;

  };


  /**
   * \brief Deduction guide
   */
  template<typename Arg>
  single_view(Arg&&) -> single_view<Arg>;


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
    return std::is_eq(operator<=>(lhs, rhs));
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

} // namespace OpenKalman


#ifdef __cpp_lib_ranges
  namespace std::ranges
#else
  namespace OpenKalman::ranges
#endif
  {
    template<typename T>
    constexpr bool enable_borrowed_range<OpenKalman::collections::single_view<T>> = std::is_lvalue_reference_v<T>;
  }


namespace std
{
  template<typename T>
  struct tuple_size<OpenKalman::collections::single_view<T>> : integral_constant<size_t, 1> {};

  template<size_t i, typename T>
  struct tuple_element<i, OpenKalman::collections::single_view<T>> { static_assert(i == 0); using type = T; };
} // namespace std


namespace OpenKalman::collections::views
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
   * \brief a RangeAdapterObject associated with \ref single_view.
   * \details The expression <code>views::single(arg)</code> is expression-equivalent
   * to <code>single_view(arg)</code> for any suitable \ref collection arg.
   * \sa single_view
   */
  inline constexpr detail::single_impl single;

}


#endif //OPENKALMAN_COLLECTIONS_VIEWS_SINGLE_HPP
