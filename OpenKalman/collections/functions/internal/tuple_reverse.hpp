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
#include "collections/concepts/uniformly_gettable.hpp"
#include "collections/concepts/tuple_like.hpp"
#include "collections/traits/size_of.hpp"
#include "collections/traits/collection_element.hpp"

namespace OpenKalman::collections
{
  /**
   * \brief A view of a tuple that reverses the order of a base tuple
   * \tparam T A base \ref uniformly_gettable object
   */
#ifdef __cpp_concepts
  template<uniformly_gettable T>
#else
  template<typename T>
#endif
  struct tuple_reverse_view
  {
  private:

    static constexpr auto base_size = size_of_v<T>;

  public:

#ifdef __cpp_concepts
    constexpr tuple_reverse_view() requires std::default_initializable<T> = default;
#else
    template<bool Enable = true, std::enable_if_t<Enable and stdcompat::default_initializable<T>, int> = 0>
    constexpr tuple_reverse_view() {};
#endif


#ifdef __cpp_concepts
    template<typename Arg> requires std::constructible_from<T, Arg&&>
#else
    template<typename Arg, std::enable_if_t<stdcompat::constructible_from<T, Arg&&>, int> = 0>
#endif
    explicit constexpr tuple_reverse_view(Arg&& arg) : t {std::forward<Arg>(arg)} {}


    /**
     * \brief Get element i of a \ref tuple_reverse_view
     */
#ifdef __cpp_concepts
    template<std::size_t i> requires (i < size_of_v<T>)
#else
    template<std::size_t i, std::enable_if_t<i < size_of_v<T>, int> = 0>
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
    template<std::size_t i> requires (i < size_of_v<T>)
#else
    template<std::size_t i, std::enable_if_t<i < size_of_v<T>, int> = 0>
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

}


namespace std
{
  template<typename T>
  struct tuple_size<OpenKalman::collections::tuple_reverse_view<T>> : OpenKalman::collections::size_of<T> {};


  template<std::size_t i, typename T>
  struct tuple_element<i, OpenKalman::collections::tuple_reverse_view<T>>
  {
    static_assert(i < OpenKalman::collections::size_of_v<T>);
    using type = OpenKalman::collections::collection_element_t<OpenKalman::collections::size_of_v<T> - i - 1, T>;
  };
}


namespace OpenKalman::collections
{
  /**
   * \brief Reverses the order of a \ref uniformly_gettable object.
   */
#ifdef __cpp_concepts
  template<uniformly_gettable Arg>
  constexpr tuple_like auto
#else
  template<typename Arg, std::enable_if_t<uniformly_gettable<Arg>, int> = 0>
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
  template<uniformly_gettable T> requires std::default_initializable<T>
  constexpr tuple_like auto
#else
  template<typename T, std::enable_if_t<uniformly_gettable<T> and stdcompat::default_initializable<T>, int> = 0>
  constexpr auto
#endif
  tuple_reverse()
  {
    return tuple_reverse_view<T> {};
  }

}

#endif
