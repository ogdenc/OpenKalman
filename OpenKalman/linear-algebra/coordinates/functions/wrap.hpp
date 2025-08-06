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
 * \brief Definition for \ref coordinates::wrap.
 */

#ifndef OPENKALMAN_COORDINATES_WRAP_HPP
#define OPENKALMAN_COORDINATES_WRAP_HPP

#include <type_traits>
#include <functional>
#include "collections/collections.hpp"
#include "linear-algebra/coordinates/interfaces/coordinate_descriptor_traits.hpp"
#include "linear-algebra/coordinates/concepts/descriptor.hpp"
#include "linear-algebra/coordinates/concepts/descriptor_collection.hpp"
#include "linear-algebra/coordinates/concepts/euclidean_pattern.hpp"
#include "linear-algebra/coordinates/functions/to_stat_space.hpp"
#include "linear-algebra/coordinates/functions/from_stat_space.hpp"
#include "linear-algebra/coordinates/traits/dimension_of.hpp"

namespace OpenKalman::coordinates
{
#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct wrap_trait_defined_for_impl : std::false_type {};

    template<typename T>
    struct wrap_trait_defined_for_impl<T, std::void_t<decltype(interface::coordinate_descriptor_traits<std::decay_t<T>>::wrap)>>
      : std::true_type {};

    template<typename T>
    constexpr bool wrap_trait_defined_for = detail::wrap_trait_defined_for_impl<T>::value;
  }
#endif


  /**
   * \brief wraps a range reflecting vector-space data to its primary range.
   * \details The wrapping operation is equivalent to mapping from modular space to Euclidean space and then back again,
   * or in other words, performing <code>to_stat_space</code> followed by <code>from_stat_space<code>.
   * \param t A \ref coordinates::descriptor "descriptor".
   * \param data_view A range within a data object corresponding to \ref coordinates::descriptor "descriptor" t.
   */
#ifdef __cpp_concepts
  template<descriptor T, collections::collection R>
  constexpr collections::collection_view decltype(auto)
#else
  template<typename T, typename R, std::enable_if_t<descriptor<T> and collections::collection<R>, int> = 0>
  constexpr decltype(auto)
#endif
  wrap(const T& t, R&& data_view)
  {
    if constexpr (dimension_of_v<T> != dynamic_size and collections::size_of_v<R> != dynamic_size)
      static_assert(dimension_of_v<T> == collections::size_of_v<R>);

    if constexpr (euclidean_pattern<T>)
    {
      return collections::views::all(std::forward<R>(data_view));
    }
#ifdef __cpp_concepts
    else if constexpr (requires { interface::coordinate_descriptor_traits<std::decay_t<T>>::wrap; })
#else
    else if constexpr (detail::wrap_trait_defined_for<T>)
#endif
    {
      auto wrap = interface::coordinate_descriptor_traits<T>::wrap;
      return collections::views::all(stdcompat::invoke(wrap, t, std::forward<R>(data_view)));
    }
    else
    {
      return from_stat_space(t, to_stat_space(t, std::forward<R>(data_view)));
    }
  }


  namespace detail
  {
    template<std::size_t t_i = 0, std::size_t data_view_i = 0, typename T, typename R, typename...Out>
    static constexpr auto
    wrap_tuple(const T& t, const R& data_view, Out...out)
    {
      if constexpr (t_i < collections::size_of_v<T>)
      {
        decltype(auto) t_elem = collections::get(t, std::integral_constant<std::size_t, t_i>{});
        auto dim = get_dimension(t_elem);
        auto data_view_sub = collections::views::slice(data_view, std::integral_constant<std::size_t, data_view_i>{}, dim);
        auto o = collections::views::all(wrap(t_elem, std::move(data_view_sub)));
        return wrap_tuple<t_i + 1, data_view_i + values::fixed_number_of_v<decltype(dim)>>(t, data_view, std::move(out)..., std::move(o));
      }
      else
      {
        return collections::views::concat(std::move(out)...);
      }
    }

  }


  /**
   * \overload
   * \brief wraps a range reflecting vector-space data to its primary range.
   * \details The wrapping operation is equivalent to mapping from modular space to Euclidean space and then back again,
   * or in other words, performing <code>to_stat_space</code> followed by <code>from_stat_space<code>.
   * \param t A \ref coordinates::descriptor_collection "descriptor_collection".
   * \param data_view A range within a data object corresponding to \ref coordinates::descriptor_collection "descriptor_collection" t.
   */
#ifdef __cpp_concepts
  template<descriptor_collection T, collections::collection R>
  constexpr collections::collection_view decltype(auto)
#else
  template<typename T, typename R, std::enable_if_t<descriptor_collection<T> and collections::collection<R>, int> = 0>
  constexpr decltype(auto)
#endif
  wrap(const T& t, R&& data_view)
  {
    if constexpr (dimension_of_v<T> != dynamic_size and collections::size_of_v<R> != dynamic_size)
      static_assert(dimension_of_v<T> == collections::size_of_v<R>);

    if constexpr (dimension_of_v<T> == dynamic_size)
    {
      std::vector<collections::common_collection_type_t<R>> data;
      data.reserve(get_dimension(t));
      std::size_t i = 0;
      for (auto& d : t)
      {
        auto dim = get_dimension(d);
#if __cpp_lib_containers_ranges >= 202002L
        data.append_range(wrap(d, collections::views::slice(data_view, i, dim)));
#else
        auto sd = wrap(d, collections::views::slice(data_view, i, dim));
        data.insert(data.end(), sd.cbegin(), sd.cend());
#endif
        i += values::to_number(dim);
      }
      return collections::views::all(std::move(data));
    }
    else //if constexpr (fixed_pattern<T>)
    {
      return detail::wrap_tuple(t, data_view);
    }
  }

} // namespace OpenKalman::coordinates


#endif