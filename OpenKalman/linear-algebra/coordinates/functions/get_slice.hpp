/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2024-2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref coordinates::get_slice.
 */

#ifndef OPENKALMAN_GET_VECTOR_SPACE_DESCRIPTOR_SLICE_HPP
#define OPENKALMAN_GET_VECTOR_SPACE_DESCRIPTOR_SLICE_HPP

#include "linear-algebra/coordinates/descriptors/internal/Slice.hpp"

namespace OpenKalman::coordinates
{
  namespace detail
  {
#ifndef __cpp_concepts
    template<typename T, typename Offset, typename Extent, typename = void>
    struct slice_is_within_range : std::false_type {};

    template<typename T, typename Offset, typename Extent>
    struct slice_is_within_range<T, Offset, Extent, std::enable_if_t<
      values::dynamic<Offset> and values::dynamic<Extent>>>
      : std::true_type {};

    template<typename T, typename Offset, typename Extent>
    struct slice_is_within_range<T, Offset, Extent, std::enable_if_t<
      fixed_pattern<T> and values::fixed<Offset> and values::dynamic<Extent>>>
      : std::bool_constant<values::fixed_number_of_v<Offset> <= dimension_of_v<T>> {};

    template<typename T, typename Offset, typename Extent>
    struct slice_is_within_range<T, Offset, Extent, std::enable_if_t<
      fixed_pattern<T> and values::dynamic<Offset> and values::fixed<Extent>>>
      : std::bool_constant<values::fixed_number_of_v<Extent> <= dimension_of_v<T>> {};

    template<typename T, typename Offset, typename Extent>
    struct slice_is_within_range<T, Offset, Extent, std::enable_if_t<
      fixed_pattern<T> and values::fixed<Offset> and values::fixed<Extent>>>
      : std::bool_constant<values::fixed_number_of_v<Offset> + values::fixed_number_of_v<Extent> <= dimension_of_v<Arg>> {};
#endif
  } // namespace detail


  /**
   * \brief Get a slice of \ref coordinates::pattern T
   * \tparam T The \ref coordinates::pattern
   * \tparam Offset The beginning location of the slice.
   * \tparam Extent The size of the slize.
   */
#ifdef __cpp_concepts
  template<coordinates::pattern Arg, values::index Offset, values::index Extent> requires dynamic_pattern<Arg> or
    ((values::dynamic<Offset> or values::fixed_number_of_v<Offset> <= dimension_of_v<Arg>) and
    (values::dynamic<Extent> or values::fixed_number_of_v<Extent> <= dimension_of_v<Arg>) and
    (values::dynamic<Offset> or values::dynamic<Extent> or values::fixed_number_of_v<Offset> + values::fixed_number_of_v<Extent> <= dimension_of_v<Arg>))
#else
  template<typename Scalar, typename Arg, typename Offset, typename Extent, std::enable_if_t<
    values::number<Scalar> and coordinates::pattern<Arg> and values::index<Offset> and values::index<Extent> and
    (dynamic_pattern<Arg> or detail::slice_is_within_range<Arg, Offset, Extent>::value), int> = 0>
#endif
  constexpr auto
  get_slice(Arg&& arg, const Offset& offset, const Extent& extent)
  {
    return internal::Slice {std::forward<Arg>(arg), offset, extent};
  }


} // namespace OpenKalman::coordinates


#endif //OPENKALMAN_GET_VECTOR_SPACE_DESCRIPTOR_SLICE_HPP
