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
 * \brief Definition for \ref collections::get_size.
 */

#ifndef OPENKALMAN_COLLECTIONS_GET_SIZE_HPP
#define OPENKALMAN_COLLECTIONS_GET_SIZE_HPP

#include "values/values.hpp"
#include "collections/concepts/sized.hpp"

namespace OpenKalman::collections
{
  namespace detail_get_size
  {
#ifndef __cpp_concepts
    template<typename T, typename = void>
    struct has_tuple_size : std::false_type {};

    template<typename T>
    struct has_tuple_size<T, std::void_t<decltype(std::tuple_size<std::decay_t<T>>::value)>> : std::true_type {};
#endif


#ifdef __cpp_concepts
    template<typename T>
#else
    template<typename T, typename = void>
#endif
    struct range_extent_impl : std::integral_constant<std::size_t, stdex::dynamic_extent> {};


#ifdef __cpp_concepts
    template<typename T> requires requires { std::tuple_size<T>::value; }
    struct range_extent_impl<T>
#else
    template<typename T>
    struct range_extent_impl<T, std::enable_if_t<has_tuple_size<T>::value>>
#endif
      : std::tuple_size<T> {};


#ifdef __cpp_concepts
    template<typename T> requires values::fixed<decltype(stdex::ranges::size(std::declval<T>()))> and
      (not requires { std::tuple_size<T>::value; })
    struct range_extent_impl<T>
#else
    template<typename T>
    struct range_extent_impl<T, std::enable_if_t<values::fixed<decltype(stdex::ranges::size(std::declval<T>()))> and
      not has_tuple_size<T>::value>>
#endif
      : values::fixed_value_of<decltype(stdex::ranges::size(std::declval<T>()))> {};


    // If T has a static size() member function (e.g., ranges::empty_view and ranges::single_view)
#ifdef __cpp_concepts
    template<typename T> requires std::bool_constant<(std::decay_t<T>::size(), true)>::value and
      (not values::fixed<decltype(stdex::ranges::size(std::declval<T>()))>) and
      (not requires { std::tuple_size<T>::value; })
    struct range_extent_impl<T>
#else
    template<typename T>
    struct range_extent_impl<T, std::enable_if_t<std::bool_constant<(std::decay_t<T>::size(), true)>::value and
      (not values::fixed<decltype(stdex::ranges::size(std::declval<T>()))>) and
      not has_tuple_size<T>::value>>
#endif
      : std::integral_constant<std::size_t, std::decay_t<T>::size()> {};


    template<typename T>
    struct range_extent : range_extent_impl<T> {};

    template<typename T>
    struct range_extent<T[]> : std::integral_constant<std::size_t, 0> {};

    template<typename T, std::size_t N>
    struct range_extent<T[N]> : std::integral_constant<std::size_t, N> {};

    //template<typename T>
    //struct range_extent<stdex::ranges::empty_view<T>> : std::integral_constant<std::size_t, 0> {};

    //template<typename T>
    //struct range_extent<stdex::ranges::single_view<T>> : std::integral_constant<std::size_t, 1> {};

    template<typename R>
    struct range_extent<stdex::ranges::ref_view<R>> : range_extent<stdex::remove_cvref_t<R>> {};

    template<typename R>
    struct range_extent<stdex::ranges::owning_view<R>> : range_extent<stdex::remove_cvref_t<R>> {};

    template<typename V, typename F>
    struct range_extent<stdex::ranges::transform_view<V, F>> : range_extent<stdex::remove_cvref_t<V>> {};

    template<typename V>
    struct range_extent<stdex::ranges::reverse_view<V>> : range_extent<stdex::remove_cvref_t<V>> {};

    template<typename...Views>
    struct range_extent<stdex::ranges::concat_view<Views...>> : std::integral_constant<std::size_t,
      (... or (range_extent<Views>::value == stdex::dynamic_extent)) ? stdex::dynamic_extent :
        (0_uz + ... + range_extent<Views>::value)> {};

#ifdef __cpp_lib_ranges
    template<typename V>
    struct range_extent<std::ranges::common_view<V>> : range_extent<stdex::remove_cvref_t<V>> {};

    template<typename V, std::size_t N>
    struct range_extent<std::ranges::elements_view<V, N>> : range_extent<stdex::remove_cvref_t<V>> {};
#endif

#ifdef __cpp_lib_ranges_as_rvalue
    template<typename V>
    struct range_extent<std::ranges::as_rvalue_view<V>> : range_extent<stdex::remove_cvref_t<V>> {};
#endif

#ifdef __cpp_lib_ranges_as_const
    template<typename V>
    struct range_extent<std::ranges::as_const_view<V>> : range_extent<stdex::remove_cvref_t<V>> {};
#endif

#ifdef __cpp_lib_ranges_enumerate
    template<typename V>
    struct range_extent<std::ranges::enumerate_view<V>> : range_extent<stdex::remove_cvref_t<V>> {};
#endif

#ifdef __cpp_lib_ranges_zip
    template<typename...Views>
      struct range_extent<std::ranges::zip_view<Views...>> : std::integral_constant<std::size_t,
        (... or (range_extent<Views>::value == stdex::dynamic_extent)) ? stdex::dynamic_extent :
          std::min({range_extent<Views>::value...})> {};

    template<typename F, typename...Views>
      struct range_extent<std::ranges::zip_transform_view<F, Views...>> : std::integral_constant<std::size_t,
        (... or (range_extent<Views>::value == stdex::dynamic_extent)) ? stdex::dynamic_extent :
          std::min({range_extent<Views>::value...})> {};

    template<typename View, std::size_t N>
      struct range_extent<std::ranges::adjacent_view<View, N>>: std::integral_constant<std::size_t,
        range_extent<View>::value == stdex::dynamic_extent ? stdex::dynamic_extent :
          range_extent<View>::value >= N ? range_extent<View>::value - N + 1 : 0> {};

    template<typename View, typename F, std::size_t N>
      struct range_extent<std::ranges::adjacent_transform_view<View, F, N>> : std::integral_constant<std::size_t,
        range_extent<View>::value == stdex::dynamic_extent ? stdex::dynamic_extent :
          range_extent<View>::value >= N ? range_extent<View>::value - N + 1 : 0> {};
#endif

#ifdef __cpp_lib_ranges_cartesian_product
    template<typename...Vs>
    struct range_extent<std::ranges::cartesian_product_view<Vs...>> : std::integral_constant<std::size_t,
      (... or (range_extent<Vs>::value == stdex::dynamic_extent)) ? stdex::dynamic_extent :
        (1_uz * ... * range_extent<Vs>::value)> {};
#endif

#ifdef __cpp_lib_ranges_cache_latest
    template<typename V>
    struct range_extent<std::ranges::cache_latest_view<V>> : range_extent<stdex::remove_cvref_t<V>> {};
#endif

#ifdef __cpp_lib_ranges_to_input
    template<typename V>
    struct range_extent<std::ranges::to_input_view<V>> : range_extent<stdex::remove_cvref_t<V>> {};
#endif


  }


  /**
   * \brief Get the size of a \ref sized object (e.g, a \ref collection)
   */
#ifdef __cpp_concepts
  template<sized Arg>
  constexpr values::index auto
#else
  template<typename Arg, std::enable_if_t<sized<Arg>, int> = 0>
  constexpr auto
#endif
  get_size(Arg&& arg)
  {
    using Ex = detail_get_size::range_extent<stdex::remove_cvref_t<Arg>>;
    if constexpr (Ex::value != stdex::dynamic_extent) { return values::cast_to<std::size_t>(Ex {}); }
    else { return static_cast<std::size_t>(stdex::ranges::size(std::forward<Arg>(arg))); }
  };


}


#endif
