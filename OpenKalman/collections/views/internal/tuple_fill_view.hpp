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
 * \brief Definition of \ref collections::internal::repeat_tuple_view.
 */

#ifndef OPENKALMAN_TUPLE_FILL_VIEW_HPP
#define OPENKALMAN_TUPLE_FILL_VIEW_HPP

#include <type_traits>
#include <tuple>

namespace OpenKalman::collections::internal
{
  /**
   * \internal
   * \brief A view of a tuple that replicates a particular value N number of times
   * \tparam N The number of copies
   * \tparam T The type of the object to be replicated
   */
  template<std::size_t N, typename T>
  struct repeat_tuple_view
  {
#ifdef __cpp_concepts
    constexpr repeat_tuple_view() requires std::default_initializable<T> = default;
#else
    template<typename aT = T, std::enable_if_t<std::is_default_constructible_v<aT>, int> = 0>
    constexpr repeat_tuple_view() {};
#endif


#ifdef __cpp_concepts
    template<typename Arg> requires std::constructible_from<T, Arg&&>
#else
    template<typename Arg, std::enable_if_t<std::is_constructible_v<T, Arg&&>, int> = 0>
#endif
    explicit constexpr repeat_tuple_view(Arg&& arg) : t {std::forward<Arg>(arg)} {}


    /**
     * \return The underlying replicated value
     */
    constexpr T value() const { return t; }


    /**
     * \brief Get element i of a \ref repeat_tuple_view
     */
#ifdef __cpp_concepts
    template<std::size_t i> requires (i < N)
#else
    template<std::size_t i, std::enable_if_t<i < N, int> = 0>
#endif
    friend constexpr T
    get(const repeat_tuple_view& v)
    {
      return v.t;
    }


    /**
    * \brief Get element i of a \ref repeat_tuple_view
    */
#ifdef __cpp_concepts
    template<size_t i> requires (i < N)
#else
    template<size_t i, std::enable_if_t<i < N, int> = 0>
#endif
    friend constexpr T
    get(repeat_tuple_view&& v)
    {
      return std::move(v).t;
    }

  private:

    T t;
  };

}


namespace std
{
  template<std::size_t N, typename T>
  struct tuple_size<OpenKalman::collections::internal::repeat_tuple_view<N, T>> : std::integral_constant<size_t, N> {};


  template<std::size_t i, std::size_t N, typename T>
  struct tuple_element<i, OpenKalman::collections::internal::repeat_tuple_view<N, T>>
  {
    static_assert(i < N);
    using type = T;
  };
} // namespace std


#endif //OPENKALMAN_TUPLE_FILL_VIEW_HPP
