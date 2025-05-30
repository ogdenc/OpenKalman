/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \internal
 * \file
 * \brief Definition of \ref coordinate_descriptor_traits.
 */

#ifndef OPENKALMAN_VECTOR_SPACE_TRAITS_HPP
#define OPENKALMAN_VECTOR_SPACE_TRAITS_HPP

#include <type_traits>
#include <typeindex>
#ifdef __cpp_lib_ranges
#include <ranges>
#else
#include "basics/compatibility/ranges.hpp"
#endif
#include "values/concepts/index.hpp"
#include "collections/concepts/collection.hpp"
#include "collections/traits/size_of.hpp"

namespace OpenKalman::interface
{
  /**
   * \brief Traits for \ref coordinates::pattern objects.
   */
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void>
#endif
  struct coordinate_descriptor_traits
  {
    static constexpr bool is_specialized = false;


    /**
     * \brief A callable object returning the number of dimensions at compile time (as a \ref values::index).
     */
    static constexpr auto dimension = [](const T& t) noexcept { return std::integral_constant<std::size_t, 0_uz>{}; };


    /**
     * \brief A callable object returning the number of dimensions after transforming to Euclidean space (as a \ref values::index).
     */
    static constexpr auto stat_dimension = [](const T& t) noexcept { return std::integral_constant<std::size_t, 0_uz>{}; };


    /**
     * \brief A callable object returning a bool reflecting whether the \ref coordinates::pattern object describes Euclidean coordinates.
     * \details In this case, dimension() == stat_dimension().
     */
    static constexpr auto is_euclidean = [](const T& t) noexcept { return std::false_type {}; };

    /**
     * \brief A unique hash code for type T, of type std::size_t.
     * \details Two coordinates will be equivalent if they have the same hash code. Generally, this can be obtained
     * through calling <code>typeid(t).hash_code()</code>.
     * \returns std::size_t
     */
    static constexpr auto hash_code = [](const T& t) noexcept -> std::size_t { return typeid(t).hash_code(); };


    /**
     * \brief A callable object mapping a range reflecting vector-space data to an corresponding range in a vector space for directional statistics.
     * \details This is the inverse of <code>from_stat_space</code>.
     * \note Optional if T is a \ref coordinates::euclidean_pattern.
     * \param data_range A range within a data object corresponding to the descriptor
     * \returns A \ref collections::collection of elements in the transformed statistical space
     */
    static constexpr auto to_stat_space =
#ifdef __cpp_concepts
    [](const T& t, collections::collection auto&& data_range) noexcept -> collections::collection decltype(auto)
#else
    [](const T& t, auto&& data_range) noexcept
#endif
    {
      if constexpr (collections::size_of_v<decltype(data_range)> != dynamic_size and values::fixed<decltype(dimension(std::declval<const T&>()))>)
        static_assert(collections::size_of_v<decltype(data_range)> == values::fixed_number_of_v<decltype(dimension(std::declval<const T&>()))>);
      return std::forward<decltype(data_range)>(data_range);
    };


    /**
     * \brief A callable object mapping a range in a vector space for directional statistics back to a range corresponding to the original vector space.
     * \details This is the inverse of <code>to_stat_space</code>.
     * \note Optional if T is a \ref coordinates::euclidean_pattern.
     * \param data_range A collection of elements within a data object in directional-statistics space corresponding to the descriptor
     * \returns A \ref collections::collection of vector elements in the original vector space
     */
    static constexpr auto from_stat_space =
#ifdef __cpp_concepts
    [](const T& t, collections::collection auto&& data_range) noexcept -> collections::collection decltype(auto)
#else
    [](const T& t, auto&& data_range) noexcept
#endif
    {
      if constexpr (collections::size_of_v<decltype(data_range)> != dynamic_size and values::fixed<decltype(stat_dimension(std::declval<const T&>()))>)
        static_assert(collections::size_of_v<decltype(data_range)> == values::fixed_number_of_v<decltype(stat_dimension(std::declval<const T&>()))>);
      return std::forward<decltype(data_range)>(data_range);
    };


    /**
     * \brief A callable object that gets a wrapped component from a range in a vector space.
     * \details The wrapped range is equivalent to <code>from_stat_space(t, to_stat_space(t, data_range))<code>.
     * \note Optional if T is a \ref coordinates::euclidean_pattern.
     * \param data_range A collection of elements within a data object corresponding to the descriptor
     * \returns A wrapped component
     */
    static constexpr auto get_wrapped_component =
#ifdef __cpp_concepts
    [](const T& t, collections::collection auto&& data_range, values::index auto i) noexcept
#else
    [](const T& t, auto&& data_range, auto i) noexcept
#endif
    {
      if constexpr (collections::size_of_v<decltype(data_range)> != dynamic_size and values::fixed<decltype(dimension(std::declval<const T&>()))>)
        static_assert(collections::size_of_v<decltype(data_range)> == values::fixed_number_of_v<decltype(dimension(std::declval<const T&>()))>);
      if constexpr (values::fixed<decltype(i)> and collections::size_of_v<decltype(data_range)> != dynamic_size)
        static_assert(values::fixed_number_of_v<decltype(i)> < collections::size_of_v<decltype(data_range)>);
      if constexpr (values::fixed<decltype(i)> and values::fixed<decltype(dimension(std::declval<const T&>()))>)
        static_assert(values::fixed_number_of_v<decltype(i)> < values::fixed_number_of_v<decltype(dimension(std::declval<const T&>()))>);
      return get(std::forward<decltype(data_range)>(data_range), std::move(i));
    };


    /**
     * \brief A callable object that sets a component from a range in a vector space, and wraps, if necessary, any components in the range.
     * \details The wrapped range is equivalent to <code>from_stat_space(t, to_stat_space(t, data_range))<code>.
     * \note Optional if T is a \ref coordinates::euclidean_pattern.
     * \param data_range A collection of elements within a data object corresponding to the descriptor
     */
    static constexpr auto set_wrapped_component =
#ifdef __cpp_concepts
    [](const T& t, collections::collection auto&& data_range, values::value auto x, values::index auto i) noexcept
      requires std::assignable_from<std::ranges::range_reference_t<decltype(data_range)>, decltype(x)&&>
#else
    [](const T& t, auto&& data_range, auto x, auto i) noexcept
#endif
    {
      if constexpr (collections::size_of_v<decltype(data_range)> != dynamic_size and values::fixed<decltype(dimension(std::declval<const T&>()))>)
        static_assert(collections::size_of_v<decltype(data_range)> == values::fixed_number_of_v<decltype(dimension(std::declval<const T&>()))>);
      if constexpr (values::fixed<decltype(i)> and collections::size_of_v<decltype(data_range)> != dynamic_size)
        static_assert(values::fixed_number_of_v<decltype(i)> < collections::size_of_v<decltype(data_range)>);
      if constexpr (values::fixed<decltype(i)> and values::fixed<decltype(dimension(std::declval<const T&>()))>)
        static_assert(values::fixed_number_of_v<decltype(i)> < values::fixed_number_of_v<decltype(dimension(std::declval<const T&>()))>);
      get(std::forward<decltype(data_range)>(data_range), std::move(i)) = std::move(x);
    };

  };


  /**
   * \brief Traits for \ref values::index.
   */
#ifdef __cpp_concepts
  template<values::index T>
  struct coordinate_descriptor_traits<T>
#else
  template<typename T>
  struct coordinate_descriptor_traits<T, std::enable_if_t<values::index<T>>>
#endif
  {
    static constexpr bool is_specialized = true;

    using scalar_type = values::number_type_of_t<T>;

    static constexpr auto dimension = [](const T& t) { return t; };

    static constexpr auto stat_dimension = [](const T& t) { return t; };

    static constexpr auto is_euclidean = [](const T&) { return std::true_type {}; };

    static constexpr auto hash_code = [](const T& t) -> std::size_t { return t; };


    static constexpr auto to_stat_space =
#ifdef __cpp_concepts
    [](const T& t, collections::collection auto&& data_range) noexcept
#else
    [](const T& t, auto&& data_range) noexcept
#endif
    {
      if constexpr (collections::size_of_v<decltype(data_range)> != dynamic_size and values::fixed<T>)
        static_assert(collections::size_of_v<decltype(data_range)> == values::fixed_number_of_v<T>);
      return std::forward<decltype(data_range)>(data_range);
    };


    static constexpr auto from_stat_space =
#ifdef __cpp_concepts
    [](const T& t, collections::collection auto&& data_range) noexcept
#else
    [](const T& t, auto&& data_range) noexcept
#endif
    {
      if constexpr (collections::size_of_v<decltype(data_range)> != dynamic_size and values::fixed<T>)
        static_assert(collections::size_of_v<decltype(data_range)> == values::fixed_number_of_v<T>);
      return std::forward<decltype(data_range)>(data_range);
    };


    static constexpr auto get_wrapped_component =
#ifdef __cpp_concepts
    [](const T& t, collections::collection auto&& data_range, values::index auto i) noexcept
#else
    [](const T& t, auto&& data_range, auto i) noexcept
#endif
    {
      if constexpr (collections::size_of_v<decltype(data_range)> != dynamic_size and values::fixed<decltype(dimension(std::declval<const T&>()))>)
        static_assert(collections::size_of_v<decltype(data_range)> == values::fixed_number_of_v<decltype(dimension(std::declval<const T&>()))>);
      if constexpr (values::fixed<decltype(i)> and collections::size_of_v<decltype(data_range)> != dynamic_size)
        static_assert(values::fixed_number_of_v<decltype(i)> < collections::size_of_v<decltype(data_range)>);
      if constexpr (values::fixed<decltype(i)> and values::fixed<decltype(dimension(std::declval<const T&>()))>)
        static_assert(values::fixed_number_of_v<decltype(i)> < values::fixed_number_of_v<decltype(dimension(std::declval<const T&>()))>);
      return get(std::forward<decltype(data_range)>(data_range), std::move(i));
    };


    static constexpr auto set_wrapped_component =
#ifdef __cpp_concepts
    [](const T& t, collections::collection auto&& data_range, values::value auto x, values::index auto i) noexcept
      requires std::assignable_from<std::ranges::range_reference_t<decltype(data_range)>, decltype(x)&&>
#else
    [](const T& t, auto&& data_range, auto x, auto i) noexcept
#endif
    {
      if constexpr (collections::size_of_v<decltype(data_range)> != dynamic_size and values::fixed<decltype(dimension(std::declval<const T&>()))>)
        static_assert(collections::size_of_v<decltype(data_range)> == values::fixed_number_of_v<decltype(dimension(std::declval<const T&>()))>);
      if constexpr (values::fixed<decltype(i)> and collections::size_of_v<decltype(data_range)> != dynamic_size)
        static_assert(values::fixed_number_of_v<decltype(i)> < collections::size_of_v<decltype(data_range)>);
      if constexpr (values::fixed<decltype(i)> and values::fixed<decltype(dimension(std::declval<const T&>()))>)
        static_assert(values::fixed_number_of_v<decltype(i)> < values::fixed_number_of_v<decltype(dimension(std::declval<const T&>()))>);
      get(std::forward<decltype(data_range)>(data_range), std::move(i)) = std::move(x);
    };

  };


} // namespace OpenKalman::interface



#endif //OPENKALMAN_VECTOR_SPACE_TRAITS_HPP
