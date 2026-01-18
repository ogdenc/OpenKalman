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

#include <variant>
#include "values/values.hpp"
#include "collections/traits/std-extents.hpp"

namespace OpenKalman::collections
{
  namespace detail_get_size
  {
#ifdef __cpp_concepts
    template<typename T>
    struct range_extent {};


    template<typename T> requires requires { std::tuple_size<T>::value; }
    struct range_extent<T> : std::tuple_size<T> {};


    template<typename T> requires values::fixed<decltype(stdex::ranges::size(std::declval<T>()))> and
      (not requires { std::tuple_size<T>::value; })
    struct range_extent<T>
      : values::fixed_value_of<decltype(stdex::ranges::size(std::declval<T>()))> {};


    // If T has a static size() member function (e.g., ranges::empty_view and ranges::single_view)
    template<typename T> requires std::bool_constant<(std::decay_t<T>::size(), true)>::value and
      (not values::fixed<decltype(stdex::ranges::size(std::declval<T>()))>) and
      (not requires { std::tuple_size<T>::value; })
    struct range_extent<T>
      : std::integral_constant<std::size_t, std::decay_t<T>::size()> {};
#else
    template<typename T, typename = void>
    struct has_tuple_size : std::false_type {};

    template<typename T>
    struct has_tuple_size<T, std::void_t<decltype(std::tuple_size<std::decay_t<T>>::value)>> : std::true_type {};


    template<typename T, typename = void>
    struct range_extent_impl {};

    template<typename T>
    struct range_extent_impl<T, std::enable_if_t<has_tuple_size<T>::value>>
      : std::tuple_size<T> {};

    template<typename T>
    struct range_extent_impl<T, std::enable_if_t<values::fixed<decltype(stdex::ranges::size(std::declval<T>()))> and
      not has_tuple_size<T>::value>>
      : values::fixed_value_of<decltype(stdex::ranges::size(std::declval<T>()))> {};

    template<typename T>
    struct range_extent_impl<T, std::enable_if_t<std::bool_constant<(std::decay_t<T>::size(), true)>::value and
      (not values::fixed<decltype(stdex::ranges::size(std::declval<T>()))>) and
      not has_tuple_size<T>::value>>
      : std::integral_constant<std::size_t, std::decay_t<T>::size()> {};


    template<typename T, typename = void>
    struct range_extent : range_extent_impl<T> {};
#endif


    template<typename T>
    constexpr std::size_t
    ext_val = []
    {
      if constexpr (values::index<range_extent<T>>) return range_extent<T>::value;
      else return 0_uz;
    }();


    template<typename IndexType, std::size_t...Extents>
    struct range_extent<stdex::extents<IndexType, Extents...>> : std::integral_constant<std::size_t, sizeof...(Extents)> {};


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
    struct range_extent<stdex::ranges::concat_view<Views...>>
      : std::conditional_t<
        (... and values::index<range_extent<Views>>),
        std::integral_constant<std::size_t, (0_uz + ... + ext_val<Views>)>,
        std::monostate> {};


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
      struct range_extent<std::ranges::zip_view<Views...>>
      : std::conditional_t<
        (... and values::index<range_extent<Views>>),
        std::integral_constant<std::size_t, std::min({ext_val<Views>...})>,
        std::monostate> {};

    template<typename F, typename...Views>
      struct range_extent<std::ranges::zip_transform_view<F, Views...>>
      : std::conditional_t<
        (... and values::index<range_extent<Views>>),
        std::integral_constant<std::size_t, std::min({ext_val<Views>...})>,
        std::monostate> {};

    template<typename View, std::size_t N>
    struct range_extent<std::ranges::adjacent_view<View, N>>
      : std::conditional_t<
        values::index<range_extent<View>>,
        std::integral_constant<std::size_t, ext_val<View> >= N ? ext_val<View> - N + 1 : 0>,
        std::monostate> {};

    template<typename View, typename F, std::size_t N>
      struct range_extent<std::ranges::adjacent_transform_view<View, F, N>>
      : std::conditional_t<
        values::index<range_extent<View>>,
        std::integral_constant<std::size_t, ext_val<View> >= N ? ext_val<View> - N + 1 : 0>,
        std::monostate> {};
#endif


#ifdef __cpp_lib_ranges_cartesian_product
    template<typename...Vs>
    struct range_extent<std::ranges::cartesian_product_view<Vs...>>
      : std::conditional_t<
        (... and values::index<range_extent<Vs>>),
        std::integral_constant<std::size_t, (1_uz * ... * ext_val<Vs>)>,
        std::monostate> {};
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
  template<typename Arg>
#ifdef __cpp_concepts
  constexpr values::size auto
#else
  constexpr auto
#endif
  get_size(Arg&& arg)
  {
    using Ex = detail_get_size::range_extent<stdex::remove_cvref_t<Arg>>;
    if constexpr (values::index<Ex>)
      return values::cast_to<std::size_t>(Ex{});
    else if constexpr (stdex::ranges::sized_range<Arg>)
      return static_cast<std::size_t>(stdex::ranges::size(std::forward<Arg>(arg)));
    else
      return values::unbounded_size;
  };


}


#endif
