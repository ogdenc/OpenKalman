/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \internal
 * \brief Definition of \ref collections::tuple_slice.
 */

#ifndef OPENKALMAN_TUPLE_SLICE_HPP
#define OPENKALMAN_TUPLE_SLICE_HPP

#include <type_traits>
#include <tuple>
#include "collections/concepts/tuple_like.hpp"

namespace OpenKalman::collections
{
  /**
   * \brief A view to a slice of a \ref tuple_like object
   * \tparam T A base \ref tuple_like object
   * \tparam index1 The index of the beginning of the slice.
   * \tparam index2 The first index just beyond the end of the slice.
   */
#ifdef __cpp_concepts
  template<std::size_t index1, std::size_t index2, tuple_like T> requires
    (index1 <= index2) and (index2 <= std::tuple_size_v<std::decay_t<T>>)
#else
  template<std::size_t index1, std::size_t index2, typename T>
#endif
  struct tuple_slice_view
  {
  private:

    static constexpr auto base_size = std::tuple_size_v<std::decay_t<T>>;

  public:

#ifdef __cpp_concepts
    constexpr tuple_slice_view() requires std::default_initializable<T> = default;
#else
    template<typename aT = T, std::enable_if_t<std::is_default_constructible_v<aT>, int> = 0>
    constexpr tuple_slice_view() {};
#endif


#ifdef __cpp_concepts
    template<typename Arg> requires std::constructible_from<T, Arg&&>
#else
    template<typename Arg, std::enable_if_t<std::is_constructible_v<T, Arg&&>, int> = 0>
#endif
    explicit constexpr tuple_slice_view(Arg&& arg) : t {std::forward<Arg>(arg)} {}


    /**
     * \brief Get element i of a \ref tuple_slice_view
     */
#ifdef __cpp_concepts
    template<std::size_t i> requires (i + index1 < index2)
#else
    template<std::size_t i, std::enable_if_t<(i + index1 < index2), int> = 0>
#endif
    friend constexpr decltype(auto)
    get(const tuple_slice_view& v)
    {
      return get(v.t, std::integral_constant<std::size_t, i + index1>{});
    }


    /**
     * \overload
     */
#ifdef __cpp_concepts
    template<std::size_t i> requires (i + index1 < index2)
#else
    template<std::size_t i, std::enable_if_t<(i + index1 < index2), int> = 0>
#endif
    friend constexpr decltype(auto)
    get(tuple_slice_view&& v)
    {
      return get(std::move(v).t, std::integral_constant<std::size_t, i + index1>{});
    }

  private:

    T t;
  };

} // namespace OpenKalman::internal


namespace std
{
  template<std::size_t index1, std::size_t index2, typename T>
  struct tuple_size<OpenKalman::collections::tuple_slice_view<index1, index2, T>> : std::integral_constant<std::size_t, index2 - index1> {};


  template<std::size_t i, std::size_t index1, std::size_t index2, typename T>
  struct tuple_element<i, OpenKalman::collections::tuple_slice_view<index1, index2, T>>
  {
    static_assert(i + index1 < index2);
    using type = std::tuple_element_t<i + index1, std::decay_t<T>>;
  };
} // namespace std


namespace OpenKalman::collections
{
  /**
   * \brief Takes a slice of a tuple, given an index range.
   * \details The function will copy or move elements from the argument tuple.
   * \tparam index1 The index of the beginning of the slice.
   * \tparam index2 The first index just beyond the end of the slice.
   * \param arg The tuple.
   * \return The tuple slice.
   */
#ifdef __cpp_concepts
  template<std::size_t index1, std::size_t index2, tuple_like Arg> requires
    (index1 <= index2) and (index2 <= std::tuple_size_v<std::remove_reference_t<Arg>>)
#else
  template<std::size_t index1, std::size_t index2, typename Arg, std::enable_if_t<tuple_like<Arg> and
    (index1 <= index2) and (index2 <= std::tuple_size<std::remove_reference_t<Arg>>::value), int> = 0>
#endif
  constexpr auto
  tuple_slice(Arg&& arg)
  {
    return tuple_slice_view<index1, index2, Arg> {std::forward<Arg>(arg)};
  }


  /**
   * \overload
   */
#ifdef __cpp_concepts
  template<std::size_t index1, std::size_t index2, tuple_like T> requires std::default_initializable<T>
#else
  template<std::size_t index1, std::size_t index2, typename T, std::enable_if_t<
    tuple_like<T> and std::is_default_constructible_v<T>, int> = 0>
#endif
  constexpr auto
  tuple_slice()
  {
    return tuple_slice_view<index1, index2, T> {};
  }

} // namespace OpenKalman::internal

#endif //OPENKALMAN_TUPLE_SLICE_HPP
