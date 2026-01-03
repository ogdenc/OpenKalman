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
 * \brief Definition for \ref collections::get_element.
 */

#ifndef OPENKALMAN_COLLECTIONS_GET_ELEMENT_HPP
#define OPENKALMAN_COLLECTIONS_GET_ELEMENT_HPP

#include "values/values.hpp"
#include "collections/traits/size_of.hpp"
#include "collections/traits/std-extents.hpp"


namespace OpenKalman::collections
{
  namespace detail_get
  {
#ifdef __cpp_concepts
    template<typename T, typename I>
#else
    template<typename T, typename I, typename = void>
#endif
    struct has_member_get_function : std::false_type {};

#ifdef __cpp_concepts
    template<typename T, typename I> requires requires { std::declval<T&&>().template get<values::fixed_value_of<I>::value>(); }
    struct has_member_get_function<T, I>
#else
    template<typename T, typename I>
    struct has_member_get_function<T, I, std::void_t<decltype(std::declval<T>().template get<values::fixed_value_of<I>::value>())>>
#endif
      : std::true_type {};


    using std::get;

#ifdef __cpp_concepts
    template<typename T, typename I>
#else
    template<typename T, typename I, typename = void>
#endif
    struct has_adl_get_function : std::false_type {};

#ifdef __cpp_concepts
    template<typename T, typename I> requires requires { get<values::fixed_value_of<I>::value>(std::declval<T&&>()); }
    struct has_adl_get_function<T, I>
#else
    template<typename T, typename I>
    struct has_adl_get_function<T, I, std::void_t<decltype(get<values::fixed_value_of<I>::value>(std::declval<T>()))>>
#endif
      : std::true_type {};


    template<typename Arg, typename I>
    constexpr decltype(auto)
    get_element_impl(Arg&& arg, I ix)
    {
      if constexpr (has_member_get_function<Arg, I>::value)
      {
        return std::forward<Arg>(arg).template get<values::fixed_value_of_v<I>>();
      }
      else if constexpr (has_adl_get_function<Arg, I>::value)
      {
        return get<values::fixed_value_of_v<I>>(std::forward<Arg>(arg));
      }
      else
      {
        std::size_t n = values::to_value_type(std::move(ix));
        if constexpr (std::is_array_v<stdex::remove_cvref_t<Arg>>)
        {
          return std::forward<Arg>(arg)[n];
        }
        else if constexpr (stdex::ranges::borrowed_range<Arg>)
        {
          return stdex::ranges::begin(std::forward<Arg>(arg))[n];
        }
        else
        {
          using namespace std;
          return begin(std::forward<Arg>(arg))[n];
        }
      }

    };

  }


  /**
   * \brief A generalization of std::get and the range subscript operator
   * \details This function works like std::get except that it works with any \ref collection.
   * The index can either be a template parameter constant or an argument. The function works as follows:
   * - If index i is a template parameter constant and if the argument has a <code>get&lt;i*gt;()</code> member, call that member.
   * - Otherwise, call <code>get&lt;i*gt;(std::forward&lt;Arg&gt;(arg))</code> if such a function is found using ADL.
   * - Otherwise, call <code>std::get&lt;i*gt;(std::forward&lt;Arg&gt;(arg))</code> if it is defined.
   * - Otherwise, if index i is either a template parameter constant or an argument,
   * call <code>std::ranges::begin(std::forward&lt;Arg&gt;(arg))</code> if it is a valid call.
   * \note This function performs no runtime bounds checking.
   */
#ifdef __cpp_concepts
  template<typename Arg, values::index I> requires
    (not values::size_compares_with<I, size_of<Arg>, &stdex::is_gteq>) and
    (stdex::ranges::random_access_range<Arg> or
      detail_get::has_member_get_function<Arg, I>::value or
      detail_get::has_adl_get_function<Arg, I>::value)
#else
  template<typename Arg, typename I, std::enable_if_t<values::index<I> and
    (not values::size_compares_with<I, size_of<Arg>, &stdex::is_gteq>) and
    (stdex::ranges::random_access_range<Arg> or
      detail_get::has_member_get_function<Arg, I>::value or
      detail_get::has_adl_get_function<Arg, I>::value), int> = 0>
#endif
  constexpr decltype(auto)
  get_element(Arg&& arg, I i)
  {
    return detail_get::get_element_impl(std::forward<Arg>(arg), i);
  }


}


#endif
