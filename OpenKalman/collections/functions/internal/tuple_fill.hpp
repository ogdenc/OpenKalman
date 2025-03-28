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
 * \brief Definition of \ref collections::tuple_fill.
 */

#ifndef OPENKALMAN_TUPLE_FILL_HPP
#define OPENKALMAN_TUPLE_FILL_HPP

#include <type_traits>
#include <tuple>

namespace OpenKalman::collections
{
  /**
   * \internal
   * \brief A view of a tuple that replicates a particular value N number of times
   * \tparam N The number of copies
   * \tparam T The type of the object to be replicated
   */
  template<std::size_t N, typename T>
  struct tuple_fill_view
  {
#ifdef __cpp_concepts
    constexpr tuple_fill_view() requires std::default_initializable<T> = default;
#else
    template<typename aT = T, std::enable_if_t<std::is_default_constructible_v<aT>, int> = 0>
    constexpr tuple_fill_view() {};
#endif


#ifdef __cpp_concepts
    template<typename Arg> requires std::constructible_from<T, Arg&&>
#else
    template<typename Arg, std::enable_if_t<std::is_constructible_v<T, Arg&&>, int> = 0>
#endif
    explicit constexpr tuple_fill_view(Arg&& arg) : t {std::forward<Arg>(arg)} {}


    /**
     * \return The underlying replicated value
     */
    constexpr T value() const { return t; }


    /**
     * \brief Get element i of a \ref tuple_fill_view
     */
#ifdef __cpp_concepts
    template<std::size_t i> requires (i < N)
#else
    template<std::size_t i, std::enable_if_t<i < N, int> = 0>
#endif
    friend constexpr T
    get(const tuple_fill_view& v)
    {
      return v.t;
    }


    /**
    * \brief Get element i of a \ref tuple_fill_view
    */
#ifdef __cpp_concepts
    template<size_t i> requires (i < N)
#else
    template<size_t i, std::enable_if_t<i < N, int> = 0>
#endif
    friend constexpr T
    get(tuple_fill_view&& v)
    {
      return std::move(v).t;
    }

  private:

    T t;
  };

} // namespace OpenKalman::internal


namespace std
{
  template<std::size_t N, typename T>
  struct tuple_size<OpenKalman::collections::tuple_fill_view<N, T>> : std::integral_constant<size_t, N> {};


  template<std::size_t i, std::size_t N, typename T>
  struct tuple_element<i, OpenKalman::collections::tuple_fill_view<N, T>>
  {
    static_assert(i < N);
    using type = T;
  };
} // namespace std


namespace OpenKalman::collections
{
  /**
   * \internal
   * \brief Creates a tuple that replicates a value N number of times.
   * \details If the argument is an lvalue reference, the result will be a tuple of lvalue references.
   * \tparam N The number of times to replicate.
   * \param arg The object to be replicated.
   * \return A \ref tuple_like object containing N copies of t.
   */
  template<std::size_t N, typename Arg>
  constexpr auto
  tuple_fill(Arg&& arg)
  {
    return tuple_fill_view<N, Arg> {std::forward<Arg>(arg)};
  }


  /**
   * \overload
   */
#ifdef __cpp_concepts
  template<std::size_t N, std::default_initializable T>
#else
  template<std::size_t N, typename T, std::enable_if_t<std::is_default_constructible_v<T>, int> = 0>
#endif
  constexpr auto
  tuple_fill()
  {
    return tuple_fill_view<N, T> {};
  }

} // namespace OpenKalman::collections

#endif //OPENKALMAN_TUPLE_FILL_HPP
