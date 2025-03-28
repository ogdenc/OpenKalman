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
 * \brief Definition for \ref collections::get.
 */

#ifndef OPENKALMAN_COLLECTIONS_GET_HPP
#define OPENKALMAN_COLLECTIONS_GET_HPP

#include <tuple>
#include "values/concepts/fixed.hpp"
#include "values/concepts/index.hpp"
#include "values/traits/fixed_number_of.hpp"
#include "collections/concepts/collection.hpp"

namespace OpenKalman::collections
{
#ifndef __cpp_lib_ranges
  namespace detail
  {
    template<std::size_t i, typename T, typename = void>
    struct internal_get_is_defined : std::false_type {};

    template<std::size_t i, typename T>
    struct internal_get_is_defined<i, T, std::void_t<decltype(OpenKalman::internal::generalized_std_get<i>(std::declval<T>()))>> : std::true_type {};
  }
#endif


  /**
   * \brief A generalization of std::get
   * \details This function takes a \ref value::value parameter instead of a template parameter like std::get.
   * - If the argument has a <code>get()</code> member, call that member.
   * - Otherwise, call <code>get&lt;i*gt;(std::forward&lt;Arg&gt;(arg))</code> if such a function is found using ADL.
   * - Otherwise, call <code>std::get&lt;i*gt;(std::forward&lt;Arg&gt;(arg))</code> if it is defined.
   */
#ifdef __cpp_lib_ranges
  template<tuple_like Arg, value::index I> requires value::fixed<I> and
    (value::fixed_number_of<I>::value < std::tuple_size<std::decay_t<Arg>>::value)
#else
  template<typename Arg, typename I, std::enable_if_t<tuple_like<Arg> and value::index<I> and
    (value::fixed_number_of<I>::value < std::tuple_size<std::decay_t<Arg>>::value), int> = 0>
#endif
  constexpr decltype(auto)
  get(Arg&& arg, const I i)
  {
    return internal::generalized_std_get<value::fixed_number_of_v<I>>(std::forward<Arg>(arg));
  }


  /**
   * \overload
   * \brief Case where the argument is a \ref sized_random_access_range and the index is \ref value::fixed "fixed".
   * \details Equivalent to <code>*(std::ranges::begin(std::forward&lt;Arg&gt;(arg)[i])</code>.
   */
#ifdef __cpp_lib_ranges
  template<sized_random_access_range Arg, value::index I> requires value::fixed<I> and (not tuple_like<Arg>)
#else
  template<typename Arg, typename I, std::enable_if_t<
    sized_random_access_range<Arg> and value::index<I> and value::fixed<I> and (not tuple_like<Arg>), int> = 0>
#endif
  constexpr decltype(auto)
  get(Arg&& arg, const I i)
  {
    using namespace std;
    if constexpr (ranges::borrowed_range<Arg>)
      return ranges::begin(std::forward<Arg>(arg))[static_cast<std::size_t>(value::fixed_number_of_v<I>)];
    else // make a copy
      { auto ret = ranges::begin(arg)[static_cast<std::size_t>(value::fixed_number_of_v<I>)]; return ret; }
  }


  /**
   * \overload
   * \brief Generalization of std::get for a \ref sized_random_access_range, where the index is \ref value::dynamic "dynamic".
   */
#ifdef __cpp_lib_ranges
  template<sized_random_access_range Arg, value::index I> requires value::dynamic<I>
#else
  template<typename Arg, typename I, std::enable_if_t<
    sized_random_access_range<Arg> and value::index<I> and value::dynamic<I>, int> = 0>
#endif
  constexpr decltype(auto)
  get(Arg&& arg, const I i)
  {
#ifdef __cpp_lib_ranges
    namespace ranges = std::ranges;
#endif
    if constexpr (ranges::borrowed_range<Arg>)
      return ranges::begin(std::forward<Arg>(arg))[static_cast<std::size_t>(i)];
    else // make a copy
      return internal::decay_copy(begin(arg)[static_cast<std::size_t>(i)]);
  };


  namespace detail
  {
    template<typename Common, typename Tup, std::size_t i>
    constexpr Common tuple_table_get(Tup&& tup)
    {
      return static_cast<Common>(collections::get(std::forward<Tup>(tup), std::integral_constant<std::size_t, i>{}));
    }


    template<typename Common, typename Tup, typename = std::make_index_sequence<std::tuple_size_v<std::decay_t<Tup>>>>
    struct tuple_table {};

    template<typename Common, typename Tup, size_t...Ix>
    struct tuple_table<Common, Tup, std::index_sequence<Ix...>>
    {
      static constexpr std::array table { tuple_table_get<Common, Tup, Ix>... };
    };

  } // namespace detail


  /**
   * \overload
   * \brief Generalization of std::get for a \ref tuple_like object, where the index is \ref value::dynamic "dynamic".
   * \tparam Common A common type to which the result of each tuple element will be converted.
   */
#ifdef __cpp_concepts
  template<typename Common, tuple_like Arg, value::index I> requires value::dynamic<I> and
    (not sized_random_access_range<Arg>) and requires { detail::tuple_table<Common, Arg>::table; }
#else
  template<typename Common, typename Arg, typename I,
    std::enable_if_t<tuple_like<Arg> and value::index<I> and value::dynamic<I> and (not sized_random_access_range<Arg>), int> = 0,
    typename = std::void_t<decltype(std::tuple_size<std::decay_t<Arg>>::value, detail::tuple_table<Common, Arg>::table)>>
#endif
  constexpr decltype(auto)
  get(Arg&& arg, const I i)
  {
    return (*detail::tuple_table<Common, Arg>::table[i])(std::forward<Arg>(arg));
  };


  namespace detail
  {
    template<typename Tup, typename = std::make_index_sequence<std::tuple_size_v<std::decay_t<Tup>>>>
    struct common_tuple_type;

    template<typename Tup, std::size_t...Ix>
    struct common_tuple_type<Tup, std::index_sequence<Ix...>>
      : std::common_type<std::tuple_element_t<Ix, std::decay_t<Tup>>...> {};
  }


  /**
   * \overload
   * \brief Generalization of std::get for a \ref tuple_like object, where the index is \ref value::dynamic "dynamic".
   * \details Unlike the previous overload, the common type will be derived, if possible, from the types of the elements.
   */
#ifdef __cpp_concepts
  template<tuple_like Arg, value::index I> requires value::dynamic<I> and (not sized_random_access_range<Arg>) and
    requires { typename detail::common_tuple_type<Arg>::type; }
#else
  template<typename Arg, typename I,
    std::enable_if_t<tuple_like<Arg> and value::index<I> and value::dynamic<I> and (not sized_random_access_range<Arg>), int> = 0,
    typename = std::void_t<decltype(std::tuple_size<std::decay_t<Arg>>::value),
      decltype(detail::tuple_table<typename detail::common_tuple_type<Arg>::type, Arg>::table)>>
#endif
  constexpr decltype(auto)
  get(Arg&& arg, const I i)
  {
    using C = typename detail::common_tuple_type<Arg>::type;
    return get<C>(std::forward<Arg>(arg), std::move(i));
  };


} // namespace OpenKalman::collections

#endif //OPENKALMAN_COLLECTIONS_GET_HPP
