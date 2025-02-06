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
 * \brief Definition for \ref value::internal::transform_collection.
 */

#ifndef OPENKALMAN_TRANSFORM_COLLECTION_HPP
#define OPENKALMAN_TRANSFORM_COLLECTION_HPP

#ifdef __cpp_lib_ranges
#include <ranges>
#else
#include <iterator>
#endif
#include "basics/internal/collection.hpp"

namespace OpenKalman::value::internal
{
#ifdef __cpp_concepts
  template<OpenKalman::internal::tuple_like C, typename F>
#else
  template<typename C, typename F>
#endif
  struct TransformTuple
  {
    explicit constexpr TransformTuple(C&& c, F&& f) : my_c {std::forward<C>(c)}, my_f {std::forward<F>(f)} {}
    template<std::size_t i> constexpr auto get() const { using std::get; return my_f(get<i>(my_c)); };

  private:

    C my_c;
    F my_f;
  };


  template<size_t i, typename C, typename F>
  constexpr auto
  get(const TransformTuple<C, F>& t) { return t.template get<i>(); };


  template<typename C, typename F>
  TransformTuple(C&&, F&&) -> TransformTuple<C, F>;

} // OpenKalman::value::internal


namespace std
{
  template<typename C, typename F>
  struct tuple_size<OpenKalman::value::internal::TransformTuple<C, F>> : std::tuple_size<std::decay_t<C>> {};

  template<size_t i, typename C, typename F>
  struct tuple_element<i, OpenKalman::value::internal::TransformTuple<C, F>>
  {
    using type = decltype(get<i>(std::declval<OpenKalman::value::internal::TransformTuple<C, F>>()));
  };


} // namespace std


namespace OpenKalman::value::internal
{
#ifndef __cpp_lib_ranges
  template<typename C, typename F>
  struct TransformRange
  {
    template<typename It>
    struct Iterator
    {
      using iterator_category = std::random_access_iterator_tag;
      using difference_type = std::ptrdiff_t;
      using value_type = decltype(std::declval<F>()(*std::declval<It>()));
      constexpr Iterator(const std::remove_reference_t<F>& f, const It& it) : my_f{f}, c_it {it} {}
      constexpr Iterator() = default;
      constexpr Iterator(const Iterator& other) = default;
      constexpr Iterator(Iterator&& other) noexcept = default;
      constexpr Iterator& operator=(const Iterator& other) = default;
      constexpr Iterator& operator=(Iterator&& other) noexcept = default;
      constexpr value_type operator*() const { return my_f(*c_it); }
      constexpr auto& operator++() noexcept { ++c_it; return *this; }
      constexpr auto operator++(int) noexcept { auto temp = *this; ++*this; return temp; }
      constexpr auto& operator--() noexcept { --c_it; return *this; }
      constexpr auto operator--(int) noexcept { auto temp = *this; --*this; return temp; }
      constexpr auto& operator+=(const difference_type diff) noexcept { c_it += diff; return *this; }
      constexpr auto& operator-=(const difference_type diff) noexcept { c_it -= diff; return *this; }
      constexpr auto operator+(const difference_type diff) const noexcept { return Iterator {my_f, c_it + diff}; }
      friend constexpr auto operator+(const difference_type diff, const Iterator& it) noexcept { return Iterator {it.my_f, diff + it.c_it}; }
      constexpr auto operator-(const difference_type diff) const noexcept { return Iterator {my_f, c_it - diff}; }
      constexpr value_type operator[](difference_type offset) const { return my_f(*(c_it + offset)); }
      constexpr bool operator==(const Iterator& other) const noexcept { return c_it == other.c_it; }
#ifdef __cpp_impl_three_way_comparison
      constexpr auto operator<=>(const Iterator& other) const noexcept { return c_it <=> other.c_it; }
#else
      constexpr bool operator!=(const Iterator& other) const noexcept { return c_it != other.c_it; }
      constexpr bool operator<(const Iterator& other) const noexcept { return c_it < other.c_it; }
      constexpr bool operator>(const Iterator& other) const noexcept { return c_it > other.c_it; }
      constexpr bool operator<=(const Iterator& other) const noexcept { return c_it <= other.c_it; }
      constexpr bool operator>=(const Iterator& other) const noexcept { return c_it >= other.c_it; }
#endif

    private:

      const std::remove_reference_t<F>& my_f;
      It c_it;

    }; // struct Iterator

    template<typename It> Iterator(F&, const It&) -> Iterator<It>;

    explicit constexpr TransformRange(C&& c, F&& f) : my_c {std::forward<C>(c)}, my_f {std::forward<F>(f)} {}
    [[nodiscard]] constexpr std::size_t size() const { using std::size; return size(my_c); }
    [[nodiscard]] constexpr auto begin() const { using std::begin; return Iterator {my_f, begin(my_c)}; }
    [[nodiscard]] constexpr auto end() const { using std::end; return Iterator {my_f, end(my_c)}; }

  private:

    C my_c;
    F my_f;

  }; // struct TransformRange


  template<typename C, typename F>
  TransformRange(C&&, F&&) -> TransformRange<C, F>;

#endif


  namespace detail
  {
#ifdef __cpp_lib_ranges
    template<OpenKalman::internal::sized_random_access_range T, typename F>
    struct is_invocable_on_range
      : std::bool_constant<std::invocable<F&, std::ranges::range_value_t<T>>> {};
#else
    template<typename T, typename F, typename = void>
    struct is_invocable_on_range : std::false_type {};

    template<typename T, typename F>
    constexpr bool
    is_invocable_on_range_impl()
    {
      using std::begin;
      return std::is_invocable_v<F&, decltype(*begin(std::declval<T>()))>;
    }

    template<typename T, typename F>
    struct is_invocable_on_range<T, F, std::enable_if_t<OpenKalman::internal::sized_random_access_range<T>>>
      : std::bool_constant<is_invocable_on_range_impl<T, F>()> {};
#endif


    template<typename Tup, typename F, std::size_t... I>
    constexpr bool
    is_invocable_on_tuple_impl(std::index_sequence<I...>)
    {
      return (... and std::is_invocable_v<F&, std::tuple_element_t<I, Tup>>);
    }


#ifdef __cpp_lib_ranges
    template<OpenKalman::internal::tuple_like T, typename F>
    struct is_invocable_on_tuple
#else
    template<typename T, typename F, typename = void>
    struct is_invocable_on_tuple : std::false_type {};

    template<typename T, typename F>
    struct is_invocable_on_tuple<T, F, std::enable_if_t<OpenKalman::internal::tuple_like<T>>>
#endif
      : std::bool_constant<is_invocable_on_tuple_impl<T, F>(std::make_index_sequence<std::tuple_size_v<T>>{})> {};

  } // namespace detail



  /**
   * \brief Transform one a \ref internal::collection "collection" to another of the same size.
   * \tparam C An underlying collection to be transformed
   * \tparam F A callable object taking an element of C and resulting in another object
   */
#ifdef __cpp_lib_ranges
  template<OpenKalman::internal::collection C, typename F> requires
    detail::is_invocable_on_range<std::decay_t<C>, std::decay_t<F>>::value or
    detail::is_invocable_on_tuple<std::decay_t<C>, std::decay_t<F>>::value
#else
  template<typename C, typename F, std::enable_if_t<OpenKalman::internal::collection<C> and
  (detail::is_invocable_on_range<std::decay_t<C>, std::decay_t<F>>::value or
    detail::is_invocable_on_tuple<std::decay_t<C>, std::decay_t<F>>::value), int> = 0>
#endif
  constexpr auto
  transform_collection(C&& c, F&& f)
  {
    if constexpr (OpenKalman::internal::sized_random_access_range<C>)
    {
#ifdef __cpp_lib_ranges
      return std::views::transform(std::forward<C>(c), std::forward<F>(f));
#else
      return TransformRange {std::forward<C>(c), std::forward<F>(f)};
#endif
    }
    else
    {
      return TransformTuple {std::forward<C>(c), std::forward<F>(f)};
    }
  };


} // namespace OpenKalman::value::internal

#endif //OPENKALMAN_TRANSFORM_COLLECTION_HPP
