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
 * \brief Definition for \ref coordinates::to_stat_space.
 */

#ifndef OPENKALMAN_COLLECTIONS_TO_STAT_SPACE_HPP
#define OPENKALMAN_COLLECTIONS_TO_STAT_SPACE_HPP

#include <functional>
#include "collections/collections.hpp"
#include "linear-algebra/coordinates/interfaces/coordinate_descriptor_traits.hpp"
#include "linear-algebra/coordinates/concepts/descriptor.hpp"
#include "linear-algebra/coordinates/concepts/descriptor_collection.hpp"
#include "linear-algebra/coordinates/concepts/euclidean_pattern.hpp"
#include "linear-algebra/coordinates/traits/dimension_of.hpp"
#include "linear-algebra/coordinates/functions/get_dimension.hpp"
#include "linear-algebra/coordinates/functions/get_stat_dimension.hpp"

namespace OpenKalman::coordinates
{
  /**
   * \brief Maps a range reflecting vector-space data to a corresponding range in a vector space for directional statistics.
   * \details This is the inverse of <code>from_stat_space</code>.
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
  to_stat_space(const T& t, R&& data_view)
  {
    if constexpr (dimension_of_v<T> != dynamic_size and collections::size_of_v<R> != dynamic_size)
      static_assert(dimension_of_v<T> == collections::size_of_v<R>);

    if constexpr (euclidean_pattern<T>)
    {
      return collections::views::all(std::forward<R>(data_view));
    }
    else
    {
      auto to_stat_space = interface::coordinate_descriptor_traits<T>::to_stat_space;
      return collections::views::all(stdcompat::invoke(to_stat_space, t, std::forward<R>(data_view)));
    }
  }


  namespace detail
  {
    template<std::size_t t_i = 0, std::size_t data_view_i = 0, typename T, typename R, typename...Out>
    static constexpr auto
    to_stat_space_tuple(const T& t, const R& data_view, Out...out)
    {
      if constexpr (t_i < collections::size_of_v<T>)
      {
        decltype(auto) t_elem = collections::get(t, std::integral_constant<std::size_t, t_i>{});
        auto dim = get_dimension(t_elem);
        auto data_view_sub = collections::views::slice(data_view, std::integral_constant<std::size_t, data_view_i>{}, dim);
        auto o = collections::views::all(to_stat_space(t_elem, std::move(data_view_sub)));
        return to_stat_space_tuple<t_i + 1, data_view_i + values::fixed_number_of_v<decltype(dim)>>(
          t, data_view, std::move(out)..., std::move(o));
      }
      else
      {
        return collections::views::concat(std::move(out)...);
      }
    }
  } // namespace detail


  /**
   * \overload
   * \brief Maps a range reflecting vector-space data to a corresponding range in a vector space for directional statistics.
   * \details This is the inverse of <code>from_stat_space</code>.
   * \param t A \ref coordinates::descriptor_collection "descriptor_collection".
   * \param data_view A range within a data object corresponding to \ref coordinates::descriptor_collection "descriptor_collection" t
   */
#ifdef __cpp_concepts
  template<descriptor_collection T, collections::collection R>
  constexpr collections::collection_view decltype(auto)
#else
  template<typename T, typename R, std::enable_if_t<descriptor_collection<T> and collections::collection<R>, int> = 0>
  constexpr decltype(auto)
#endif
  to_stat_space(const T& t, R&& data_view)
  {
    if constexpr (dimension_of_v<T> != dynamic_size and collections::size_of_v<R> != dynamic_size)
      static_assert(dimension_of_v<T> == collections::size_of_v<R>);

    if constexpr (dimension_of_v<T> == dynamic_size) 
    {
      std::vector<collections::common_collection_type_t<R>> stat_data;
      stat_data.reserve(get_stat_dimension(t));
      std::size_t i = 0;
      for (auto& d : t)
      {
        auto dim = get_dimension(d);
        auto sd = to_stat_space(d, collections::views::slice(data_view, i, dim));
        stat_data.insert(stat_data.end(), sd.cbegin(), sd.cend());
        i += values::to_number(dim);
      }
      return collections::views::all(std::move(stat_data));
    }
    else //if constexpr (fixed_pattern<T>)
    {
      return detail::to_stat_space_tuple(t, data_view);
    }
  }

} // namespace OpenKalman::coordinates


#endif
