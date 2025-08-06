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
 * \brief Definition for \ref collections::from_stat_space.
 */

#ifndef OPENKALMAN_COLLECTIONS_FROM_STAT_SPACE_HPP
#define OPENKALMAN_COLLECTIONS_FROM_STAT_SPACE_HPP

#include <functional>
#include "collections/collections.hpp"
#include "linear-algebra/coordinates/interfaces/coordinate_descriptor_traits.hpp"
#include "linear-algebra/coordinates/concepts/descriptor.hpp"
#include "linear-algebra/coordinates/concepts/descriptor_collection.hpp"
#include "linear-algebra/coordinates/concepts/euclidean_pattern.hpp"
#include "linear-algebra/coordinates/traits/stat_dimension_of.hpp"
#include "linear-algebra/coordinates/functions/get_dimension.hpp"
#include "linear-algebra/coordinates/functions/get_stat_dimension.hpp"

namespace OpenKalman::coordinates
{
  /**
   * \brief Maps a range in a vector space for directional-statistics back to a range reflecting vector-space data.
   * \details This is the inverse of <code>to_stat_space</code>.
   * \param t A \ref coordinates::descriptor "descriptor".
   * \param stat_data_view A range within a data object corresponding to \ref coordinates::descriptor "descriptor" t.
   */
#ifdef __cpp_concepts
  template<descriptor T, collections::collection R>
  constexpr collections::collection_view decltype(auto)
#else
  template<typename T, typename R, std::enable_if_t<descriptor<T> and collections::collection<R>, int> = 0>
  constexpr decltype(auto)
#endif
  from_stat_space(const T& t, R&& stat_data_view)
  {
    if constexpr (stat_dimension_of_v<T> != dynamic_size and collections::size_of_v<R> != dynamic_size)
      static_assert(stat_dimension_of_v<T> == collections::size_of_v<R>);

    if constexpr (euclidean_pattern<T>)
    {
      return collections::views::all(std::forward<R>(stat_data_view));
    }
    else
    {
      auto from_stat_space = interface::coordinate_descriptor_traits<T>::from_stat_space;
      return collections::views::all(stdcompat::invoke(from_stat_space, t, std::forward<R>(stat_data_view)));
    }
  }


  namespace detail
  {
    template<std::size_t t_i = 0, std::size_t stat_data_view_i = 0, typename T, typename R, typename...Out>
    static constexpr auto
    from_stat_space_tuple(const T& t, const R& stat_data_view, Out...out)
    {
      if constexpr (t_i < collections::size_of_v<T>)
      {
        decltype(auto) t_elem = collections::get(t, std::integral_constant<std::size_t, t_i>{});
        auto dim = get_stat_dimension(t_elem);
        auto data_view_sub = collections::views::slice(stat_data_view, std::integral_constant<std::size_t, stat_data_view_i>{}, dim);
        auto o = collections::views::all(from_stat_space(t_elem, std::move(data_view_sub)));
        return from_stat_space_tuple<t_i + 1, stat_data_view_i + values::fixed_number_of_v<decltype(dim)>>(
          t, stat_data_view, std::move(out)..., std::move(o));
      }
      else
      {
        return collections::views::concat(std::move(out)...);
      }
    }
  } // namespace detail


  /**
   * \overload
   * \brief Maps a range in a vector space for directional-statistics back to a range reflecting vector-space data.
   * \details This is the inverse of <code>to_stat_space</code>.
   * \param t A \ref coordinates::descriptor_collection "descriptor_collection".
   * \param stat_data_view A range within a data object corresponding to coordinates::descriptor_collection "descriptor_collection" t
   */
#ifdef __cpp_concepts
  template<descriptor_collection T, collections::collection R>
  constexpr collections::collection_view decltype(auto)
#else
  template<typename T, typename R, std::enable_if_t<descriptor_collection<T> and collections::collection<R>, int> = 0>
  constexpr decltype(auto)
#endif
  from_stat_space(const T& t, R&& stat_data_view)
  {
    if constexpr (stat_dimension_of_v<T> != dynamic_size and collections::size_of_v<R> != dynamic_size)
      static_assert(stat_dimension_of_v<T> == collections::size_of_v<R>);

    if constexpr (stat_dimension_of_v<T> == dynamic_size)
    {
      std::vector<collections::common_collection_type_t<R>> data;
      data.reserve(get_dimension(t));
      std::size_t i = 0;
      for (auto& d : t)
      {
        auto dim = get_stat_dimension(d);
#if __cpp_lib_containers_ranges >= 202002L
        data.append_range(to_stat_space(d, collections::views::slice(stat_data_view, i, dim)));
#else
        auto sd = from_stat_space(d, collections::views::slice(stat_data_view, i, dim));
        data.insert(data.end(), sd.cbegin(), sd.cend());
#endif
        i += values::to_number(dim);
      }
      return collections::views::all(std::move(data));
    }
    else //if constexpr (fixed_pattern<T>)
    {
      return detail::from_stat_space_tuple(t, stat_data_view);
    }
  }

} // namespace OpenKalman::coordinates


#endif
