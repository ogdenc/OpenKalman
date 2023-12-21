/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref maybe_same_shape_as.
 */

#ifndef OPENKALMAN_MAYBE_SAME_SHAPE_AS_HPP
#define OPENKALMAN_MAYBE_SAME_SHAPE_AS_HPP


namespace OpenKalman
{
  namespace detail
  {
    template<std::size_t I>
    constexpr bool maybe_dimensions_are_same() { return true; }

    template<std::size_t I, typename T, typename...Ts>
    constexpr bool maybe_dimensions_are_same()
    {
      if constexpr (dynamic_dimension<T, I>) return maybe_dimensions_are_same<I, Ts...>();
      else return ((dynamic_dimension<Ts, I> or equivalent_to<vector_space_descriptor_of_t<T, I>, vector_space_descriptor_of_t<Ts, I>>) and ...);
    }

    template<typename...Ts, std::size_t...Is>
    constexpr bool maybe_has_same_shape_as_impl(std::index_sequence<Is...>)
    {
      return (maybe_dimensions_are_same<Is, Ts...>() and ...);
    }
  } // namespace detail

  /**
   * \brief Specifies that it is not ruled out, at compile time, that T has the same dimensions and vector-space types as Ts.
   * \details Two dimensions are considered the same if their \ref vector_space_descriptor are \ref equivalent_to "equivalent".
   * \sa same_shape_as
   * \sa same_shape
   */
  template<typename...Ts>
#ifdef __cpp_concepts
  concept maybe_same_shape_as =
#else
  constexpr bool maybe_same_shape_as =
#endif
    (indexible<Ts> and ...) and
    detail::maybe_has_same_shape_as_impl<Ts...>(std::make_index_sequence<std::max({std::size_t{0}, index_count_v<Ts>...})>{});


} // namespace OpenKalman

#endif //OPENKALMAN_MAYBE_SAME_SHAPE_AS_HPP
