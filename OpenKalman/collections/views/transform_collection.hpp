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
 * \brief Definition for \ref value::transform_collection.
 */

#ifndef OPENKALMAN_TRANSFORM_COLLECTION_HPP
#define OPENKALMAN_TRANSFORM_COLLECTION_HPP

#include <type_traits>
#ifdef __cpp_lib_ranges
#include <ranges>
#else
#include "basics/ranges.hpp"
#endif
#include "collections/concepts/collection.hpp"
#include "collections/concepts/invocable_on_collection.hpp"

namespace OpenKalman::value
{
  /**
   * \brief A \ref collection created by applying a transformation to another collection of the same size.
   * \tparam C An underlying collection to be transformed
   * \tparam F A callable object taking an element of C and resulting in another object
   */
#ifdef __cpp_lib_ranges
  template<collection C, invocable_on_collection<C> F>
#else
  template<typename C, typename F, typename = void>
#endif
  struct transform_collection_view;


  /**
   * \overload
   */
#ifdef __cpp_concepts
  template<tuple_like C, invocable_on_collection<C> F> requires (not sized_random_access_range<C>)
  struct transform_collection_view<C, F>
#else
  template<typename C, typename F>
  struct transform_collection_view<C, F, std::enable_if_t<
    tuple_like<C> and invocable_on_collection<F, C> and (not sized_random_access_range<C>)>>
#endif
  {
    template<typename aC, typename aF>
    constexpr transform_collection_view(aC&& c, aF&& f) : C {std::forward<aC>(c)}, F {std::forward<aF>(f)} {}

    template<size_t i>
    constexpr auto
    get() const { return my_f(collections::get<i>(my_c)); };

  private:

    C my_c;
    F my_f;
  };


  /**
   * \overload
   */
#ifdef __cpp_lib_ranges
  template<sized_random_access_range C, invocable_on_collection<C> F> requires
    requires { typename std::ranges::transform_view<C, F>; }
  struct transform_collection_view<C, F> : std::ranges::transform_view<C, F>
  {
    constexpr transform_collection_view(C c, F f)
      : std::ranges::transform_view<C, F>{c, f}, my_c {std::move(c)}, my_f {std::move(f)} {}

    template<size_t i>
    constexpr auto
    get() const { return my_f(collections::get<i>(my_c)); };

  private:

    C my_c;
    F my_f;
  };
#endif


  /**
   * \overload
   */
#ifdef __cpp_lib_ranges
  template<sized_random_access_range C, invocable_on_collection<C> F> requires
    (not requires { typename std::ranges::transform_view<C, F>; })
  struct transform_collection_view<C, F> : std::ranges::view_interface<transform_collection_view<C, F>>
#else
  template<typename C, typename F>
  struct transform_collection_view<C, F, std::enable_if_t<sized_random_access_range<C> and invocable_on_collection<F, C>>>
#endif
  {
  private:

#ifdef __cpp_lib_ranges
    using It = std::ranges::iterator_t<C>;
#else
    using It = ranges::iterator_t<C>;
#endif

    struct Iterator
    {
      using iterator_category = std::random_access_iterator_tag;
      using value_type = decltype(std::declval<const std::remove_reference_t<F>&>()(*std::declval<It>()));
      using difference_type = std::decay_t<decltype(std::declval<value_type>() - std::declval<value_type>())>;
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

      const std::remove_reference_t<F>& my_f;
      It c_it;

    }; // struct Iterator

public:

    template<typename aC, typename aF>
    constexpr transform_collection_view(aC&& c, aF&& f) : my_c {std::forward<aC>(c)}, my_f {std::forward<aF>(f)} {}


#ifdef __cpp_concepts
    constexpr transform_collection_view() requires std::default_initializable<C> and std::default_initializable<F> = default;
#else
    template<typename aC = C, std::enable_if_t<std::is_default_constructible_v<aC> and std::is_default_constructible_v<F>, int> = 0>
    constexpr transform_collection_view() {};
#endif

#ifdef __cpp_lib_ranges
    namespace ranges = std::ranges;
#endif

    [[nodiscard]] constexpr auto begin() const { return Iterator {my_f, ranges::begin(my_c)}; }

    [[nodiscard]] constexpr auto end() const { return Iterator {my_f, ranges::end(my_c)}; }

    [[nodiscard]] constexpr auto size() const
    {
      if constexpr (tuple_like<C>) return std::tuple_size_v<C>;
      else return ranges::size(my_c);
    }

    template<size_t i>
    constexpr auto
    get() const { return my_f(collections::get<i>(my_c)); };

  private:

    C my_c;
    F my_f;

  };


  template<typename C, typename F>
  transform_collection_view(C&&, F&&) -> transform_collection_view<C, F>;


#ifdef __cpp_concepts
  template<size_t i, typename C, typename F> requires tuple_like<C>
#else
  template<size_t i, typename C, typename F, std::enable_if_t<tuple_like<C>, int> = 0>
#endif
  constexpr auto
  get(const transform_collection_view<C, F>& t)
  {
    return t.template get<i>();
  };


  /**
   * \brief Create a \ref transform_collection_view.
   * \tparam C An underlying collection to be transformed
   * \tparam F A callable object taking an element of C and resulting in another object
   */
#ifdef __cpp_lib_ranges
  template<collection C, invocable_on_collection<C> F>
  constexpr collection auto
#else
  template<typename C, typename F, std::enable_if_t<
    collection<C> and invocable_on_collection<F, C>, int> = 0>
  constexpr auto
#endif
  transform_collection(C&& c, F&& f)
  {
    return transform_collection_view {std::forward<C>(c), std::forward<F>(f)};
  };


#ifndef __cpp_concepts
  namespace detail
  {
    template<typename C, typename = void>
    struct transform_collection_view_tuple_size_impl {};

    template<typename C>
    struct transform_collection_view_tuple_size_impl<C, std::enable_if_t<tuple_like<C>>>
      : std::tuple_size<std::decay_t<C>> {};


    template<std::size_t i, typename C, typename F, typename = void>
    struct transform_collection_view_tuple_element_impl {};

    template<std::size_t i, typename C, typename F>
    struct transform_collection_view_tuple_element_impl<i, C, F, std::enable_if_t<tuple_like<C>>>
    {
      using type = std::invoke_result_t<F, std::tuple_element_t<i, C>>;
    };
  } // namespace detail
#endif

} // namespace OpenKalman::value


namespace std
{
#ifdef __cpp_concepts
  template<OpenKalman::collections::tuple_like C, typename F>
  struct tuple_size<OpenKalman::value::transform_collection_view<C, F>> : std::tuple_size<std::decay_t<C>> {};
#else
  template<typename C, typename F>
  struct tuple_size<OpenKalman::value::transform_collection_view<C, F>>
    : OpenKalman::value::detail::transform_collection_view_tuple_size_impl<C> {};
#endif


#ifdef __cpp_concepts
  template<std::size_t i, OpenKalman::collections::tuple_like C, typename F>
  struct tuple_element<i, OpenKalman::value::transform_collection_view<C, F>>
  {
    using type = std::invoke_result_t<F, std::tuple_element_t<i, C>>;
  };
#else
  template<std::size_t i, typename C, typename F>
  struct tuple_element<i, OpenKalman::value::transform_collection_view<C, F>>
    : OpenKalman::value::detail::transform_collection_view_tuple_element_impl<i, C, F> {};
#endif


} // namespace std

#endif //OPENKALMAN_TRANSFORM_COLLECTION_HPP
