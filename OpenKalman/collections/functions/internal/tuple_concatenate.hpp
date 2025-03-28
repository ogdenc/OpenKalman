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
 * \brief Definition of \ref collections::tuple_concatenate.
 */

#ifndef OPENKALMAN_TUPLE_CONCATENATE_HPP
#define OPENKALMAN_TUPLE_CONCATENATE_HPP

#include <type_traits>
#include <tuple>
#include "collections/concepts/tuple_like.hpp"

namespace OpenKalman::collections
{
  namespace detail
  {
    template<std::size_t i>
    constexpr auto tuple_concatenate_view_indices()
    {
      return std::pair {std::integral_constant<std::size_t, 0_uz>{}, std::integral_constant<std::size_t, 0_uz>{}};
    }

    template<std::size_t i, typename T, typename...Ts>
    constexpr auto tuple_concatenate_view_indices()
    {
      constexpr std::size_t size = std::tuple_size_v<std::decay_t<T>>;
      if constexpr (i < size) return std::pair {std::integral_constant<std::size_t, 0_uz>{}, std::integral_constant<std::size_t, i>{}};
      else
      {
        auto [a, b] = tuple_concatenate_view_indices<i - size, Ts...>();
        return std::pair {std::integral_constant<std::size_t, 1_uz + decltype(a)::value>{}, b};
      }
    }
  };


  /**
   * \internal
   * \brief A view to a concatenation of some number of other \ref tuple_like object
   * \details This is similar to std::tuple_cat, but it allows for concatenation of any \ref tuple_like object.
   * \tparam Ts A set of base \ref tuple_like objects
   */
#ifdef __cpp_concepts
  template<tuple_like...Ts>
#else
  template<typename...Ts>
#endif
  struct tuple_concatenate_view
  {
#ifdef __cpp_concepts
    constexpr tuple_concatenate_view() requires (... and std::default_initializable<Ts>) = default;
#else
    template<typename Tup = std::tuple<Ts...>, std::enable_if_t<std::is_default_constructible_v<Tup>, int> = 0>
    constexpr tuple_concatenate_view() {};
#endif


#ifdef __cpp_concepts
    template<typename...Args> requires (... and std::constructible_from<Ts, Args&&>)
#else
    template<typename...Args, std::enable_if_t<(... and std::is_constructible_v<Ts, Args&&>), int> = 0>
#endif
    explicit constexpr tuple_concatenate_view(Args&&...args) : tup {std::forward<Args>(args)...} {}


    /**
     * \brief Get element i of a \ref tuple_concatenate_view
     */
#ifdef __cpp_concepts
    template<std::size_t i> requires (i < (0_uz + ... + std::tuple_size_v<std::decay_t<Ts>>))
#else
    template<std::size_t i, std::enable_if_t<(i < (0_uz + ... + std::tuple_size_v<std::decay_t<Ts>>)), int> = 0>
#endif
    friend constexpr decltype(auto)
    get(const tuple_concatenate_view& v)
    {
      auto [element, index] = std::decay_t<decltype(detail::tuple_concatenate_view_indices<i, Ts...>())>();
      return get(get(v.tup, element), index);
    }


    /**
     * \overload
     */
#ifdef __cpp_concepts
    template<std::size_t i> requires (i < (0_uz + ... + std::tuple_size_v<std::decay_t<Ts>>))
#else
    template<std::size_t i, std::enable_if_t<(i < (0_uz + ... + std::tuple_size_v<std::decay_t<Ts>>)), int> = 0>
#endif
    friend constexpr decltype(auto)
    get(tuple_concatenate_view&& v)
    {
      auto [element, index] = std::decay_t<decltype(detail::tuple_concatenate_view_indices<i, Ts...>())>();
      return get(get(std::move(v).tup, element), index);
    }

  private:

    std::tuple<Ts...> tup;
  };


  /**
   * \brief Deduction guide
   */
  template<typename...Args>
  tuple_concatenate_view(Args&&...) -> tuple_concatenate_view<Args...>;

} // namespace OpenKalman::internal


namespace std
{
  template<typename...Ts>
  struct tuple_size<OpenKalman::collections::tuple_concatenate_view<Ts...>>
    : std::integral_constant<std::size_t, (0_uz + ... + std::tuple_size_v<std::decay_t<Ts>>)> {};


  template<std::size_t i, typename...Ts>
  struct tuple_element<i, OpenKalman::collections::tuple_concatenate_view<Ts...>>
  {
    static_assert(i < (0_uz + ... + std::tuple_size_v<std::decay_t<Ts>>));
    using indices = decltype(OpenKalman::collections::detail::tuple_concatenate_view_indices<i, Ts...>());
    using element = std::tuple_element_t<0, indices>;
    using index = std::tuple_element_t<1, indices>;
    using type = std::tuple_element_t<index::value, std::decay_t<std::tuple_element_t<element::value, std::tuple<Ts...>>>>;
  };
} // namespace std


namespace OpenKalman::collections
{
  /**
   * \internal
   * \brief Concatenate some number of other \ref tuple_like object
   * \details This is similar to std::tuple_cat, but it allows for concatenation of any \ref tuple_like object.
   * \tparam index1 The index of the beginning of the slice.
   * \tparam index2 The first index just beyond the end of the slice.
   * \param arg The tuple.
   * \return The tuple slice.
   */
#ifdef __cpp_concepts
  template<tuple_like...Args>
#else
  template<typename...Args, std::enable_if_t<(... and tuple_like<Args>), int> = 0>
#endif
  constexpr auto
  tuple_concatenate(Args&&...args)
  {
    return tuple_concatenate_view {std::forward<Args>(args)...};
  }


  /**
   * \overload
   */
#ifdef __cpp_concepts
  template<tuple_like...Ts> requires (... and std::default_initializable<Ts>)
#else
  template<typename...Ts, std::enable_if_t<
    (... and (tuple_like<Ts> and std::is_default_constructible_v<Ts>)), int> = 0>
#endif
  constexpr auto
  tuple_concatenate()
  {
    return tuple_concatenate_view<Ts...> {};
  }

} // namespace OpenKalman::collections

#endif //OPENKALMAN_TUPLE_CONCATENATE_HPP
