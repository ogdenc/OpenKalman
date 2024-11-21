/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition of \ref is_square_shaped function.
 */

#ifndef OPENKALMAN_IS_SQUARE_SHAPED_HPP
#define OPENKALMAN_IS_SQUARE_SHAPED_HPP

#include<optional>


namespace OpenKalman
{

  namespace detail
  {
    template<std::size_t count, typename T, std::size_t i = 0>
    constexpr auto get_best_square_index()
    {
      if constexpr (i + 1 >= count) return i;
      else if constexpr (static_vector_space_descriptor<decltype(get_vector_space_descriptor<i>(std::declval<T>()))>) return i;
      else return get_best_square_index<count, T, i + 1>();
    }


    template<std::size_t...Is, typename T>
    constexpr auto is_square_shaped_impl(std::index_sequence<Is...>, const T& t)
    {
      constexpr std::size_t bestI = get_best_square_index<sizeof...(Is), T>();
      auto dim_bestI = get_vector_space_descriptor<bestI>(t);
      if ((... and (Is == bestI or get_vector_space_descriptor<Is>(t) == dim_bestI)))
        return std::optional {dim_bestI};
      else
        return std::optional<decltype(dim_bestI)> {};
    }
  } // namespace detail


  /**
   * \brief Determine whether an object is \ref square_shaped at runtime.
   * \details An object is square-shaped if it has the same size and \ref vector_space_descriptor type along every index
   * (excluding trailing 1D indices).
   * \tparam T A tensor or matrix
   * \return a \ref std::optional which includes the \ref vector_space_descriptor object if T is square.
   * The result is convertible to <code>bool</code>: if true, then T is square.
   * \sa square_shaped
   */
#ifdef __cpp_concepts
  template<interface::count_indices_defined_for T>
#else
  template<typename T, std::enable_if_t<interface::count_indices_defined_for<T>, int> = 0>
#endif
  constexpr auto is_square_shaped(const T& t)
  {
    if constexpr (value::static_index<decltype(count_indices(t))>)
    {
      constexpr std::size_t count = std::decay_t<decltype(count_indices(t))>::value;
      return detail::is_square_shaped_impl(std::make_index_sequence<count>{}, t);
    }
    else
    {
      auto d0 = get_vector_space_descriptor<0>(t);
      using Ret = std::optional<std::decay_t<decltype(d0)>>;
      for (std::size_t i = 1; i < count_indices(t); ++i) if (d0 != get_vector_space_descriptor(t, i)) return Ret{};
      return Ret {d0};
    }
  }


} // namespace OpenKalman

#endif //OPENKALMAN_IS_SQUARE_SHAPED_HPP
