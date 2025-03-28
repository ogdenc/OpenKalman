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
 * \brief Definition of \ref collections::tuple_reverse.
 */

#ifndef OPENKALMAN_TUPLE_REVERSE_HPP
#define OPENKALMAN_TUPLE_REVERSE_HPP

#include <type_traits>
#include <tuple>
#include "collections/concepts/tuple_like.hpp"

namespace OpenKalman::collections
{
  /**
   * \brief A view of a tuple that reverses the order of a base tuple
   * \tparam T A base \ref tuple_like object
   */
#ifdef __cpp_concepts
  template<tuple_like T>
#else
  template<typename T>
#endif
  struct tuple_reverse_view
  {
  private:

    static constexpr auto base_size = std::tuple_size_v<std::decay_t<T>>;

  public:

#ifdef __cpp_concepts
    constexpr tuple_reverse_view() requires std::default_initializable<T> = default;
#else
    template<typename aT = T, std::enable_if_t<std::is_default_constructible_v<aT>, int> = 0>
    constexpr tuple_reverse_view() {};
#endif


#ifdef __cpp_concepts
    template<typename Arg> requires std::constructible_from<T, Arg&&>
#else
    template<typename Arg, std::enable_if_t<std::is_constructible_v<T, Arg&&>, int> = 0>
#endif
    explicit constexpr tuple_reverse_view(Arg&& arg) : t {std::forward<Arg>(arg)} {}


    /**
     * \brief Get element i of a \ref tuple_reverse_view
     */
#ifdef __cpp_concepts
    template<std::size_t i> requires (i < std::tuple_size_v<std::decay_t<T>>)
#else
    template<std::size_t i, std::enable_if_t<i < std::tuple_size_v<std::decay_t<T>>, int> = 0>
#endif
    friend constexpr decltype(auto)
    get(const tuple_reverse_view& v)
    {
      return collections::get(v.t, std::integral_constant<std::size_t, base_size - i - 1_uz>{});
    }


    /**
     * \overload
     */
#ifdef __cpp_concepts
    template<std::size_t i> requires (i < std::tuple_size_v<std::decay_t<T>>)
#else
    template<std::size_t i, std::enable_if_t<i < std::tuple_size_v<std::decay_t<T>>, int> = 0>
#endif
    friend constexpr decltype(auto)
    get(tuple_reverse_view&& v)
    {
      return collections::get(std::move(v).t, std::integral_constant<std::size_t, base_size - i - 1_uz>{});
    }

  private:

    T t;
  };


  /**
   * \brief Deduction guide
   */
  template<typename Arg>
  tuple_reverse_view(Arg&&) -> tuple_reverse_view<Arg>;

} // namespace OpenKalman::internal


namespace std
{
  template<typename T>
  struct tuple_size<OpenKalman::collections::tuple_reverse_view<T>> : std::tuple_size<std::decay_t<T>> {};


  template<std::size_t i, typename T>
  struct tuple_element<i, OpenKalman::collections::tuple_reverse_view<T>>
  {
    static_assert(i < std::tuple_size_v<std::decay_t<T>>);
    using type = std::tuple_element_t<std::tuple_size_v<std::decay_t<T>> - i - 1_uz, std::decay_t<T>>;
  };
} // namespace std


namespace OpenKalman::collections
{
  /**
   * \brief Reverses the order of a \ref tuple_like object.
   */
#ifdef __cpp_concepts
  template<tuple_like Arg>
  constexpr tuple_like auto
#else
  template<typename Arg, std::enable_if_t<tuple_like<Arg>, int> = 0>
  constexpr auto
#endif
  tuple_reverse(Arg&& arg)
  {
    return tuple_reverse_view {std::forward<Arg>(arg)};
  }


  /**
   * \overload
   */
#ifdef __cpp_concepts
  template<tuple_like T> requires std::default_initializable<T>
#else
  template<typename T, std::enable_if_t<tuple_like<T> and std::is_default_constructible_v<T>, int> = 0>
#endif
  constexpr auto
  tuple_reverse()
  {
    return tuple_reverse_view<T> {};
  }

} // namespace OpenKalman::collections

#endif //OPENKALMAN_TUPLE_REVERSE_HPP
