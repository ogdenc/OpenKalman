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
 * \brief Definition for \ref descriptor::get_slice.
 */

#ifndef OPENKALMAN_GET_VECTOR_SPACE_DESCRIPTOR_SLICE_HPP
#define OPENKALMAN_GET_VECTOR_SPACE_DESCRIPTOR_SLICE_HPP

#include "linear-algebra/vector-space-descriptors/descriptors/internal/Slice.hpp"

namespace OpenKalman::descriptor
{
  namespace detail
  {
#ifndef __cpp_concepts
    template<typename T, typename Offset, typename Extent, typename = void>
    struct slice_is_within_range : std::false_type {};

    template<typename T, typename Offset, typename Extent>
    struct slice_is_within_range<T, Offset, Extent, std::enable_if_t<
      value::dynamic<Offset> and value::dynamic<Extent>>>
      : std::true_type {};

    template<typename T, typename Offset, typename Extent>
    struct slice_is_within_range<T, Offset, Extent, std::enable_if_t<
      static_vector_space_descriptor<T> and value::fixed<Offset> and value::dynamic<Extent>>>
      : std::bool_constant<value::fixed_number_of_v<Offset> <= dimension_size_of_v<T>> {};

    template<typename T, typename Offset, typename Extent>
    struct slice_is_within_range<T, Offset, Extent, std::enable_if_t<
      static_vector_space_descriptor<T> and value::dynamic<Offset> and value::fixed<Extent>>>
      : std::bool_constant<value::fixed_number_of_v<Extent> <= dimension_size_of_v<T>> {};

    template<typename T, typename Offset, typename Extent>
    struct slice_is_within_range<T, Offset, Extent, std::enable_if_t<
      static_vector_space_descriptor<T> and value::fixed<Offset> and value::fixed<Extent>>>
      : std::bool_constant<value::fixed_number_of_v<Offset> + value::fixed_number_of_v<Extent> <= dimension_size_of_v<Arg>> {};
#endif
  } // namespace detail


  /**
   * \brief Get a slice of \ref vector_space_descriptor T
   * \tparam T The \ref vector_space_descriptor
   * \tparam Offset The beginning location of the slice.
   * \tparam Extent The size of the slize.
   */
#ifdef __cpp_concepts
  template<vector_space_descriptor Arg, value::index Offset, value::index Extent> requires dynamic_vector_space_descriptor<Arg> or
    ((value::dynamic<Offset> or value::fixed_number_of_v<Offset> <= dimension_size_of_v<Arg>) and
    (value::dynamic<Extent> or value::fixed_number_of_v<Extent> <= dimension_size_of_v<Arg>) and
    (value::dynamic<Offset> or value::dynamic<Extent> or value::fixed_number_of_v<Offset> + value::fixed_number_of_v<Extent> <= dimension_size_of_v<Arg>))
#else
  template<typename Scalar, typename Arg, typename Offset, typename Extent, std::enable_if_t<
    value::number<Scalar> and vector_space_descriptor<Arg> and value::index<Offset> and value::index<Extent> and
    (dynamic_vector_space_descriptor<Arg> or detail::slice_is_within_range<Arg, Offset, Extent>::value), int> = 0>
#endif
  constexpr auto
  get_slice(Arg&& arg, const Offset& offset, const Extent& extent)
  {
    return internal::Slice {std::forward<Arg>(arg), offset, extent};
  }


} // namespace OpenKalman::descriptor


#endif //OPENKALMAN_GET_VECTOR_SPACE_DESCRIPTOR_SLICE_HPP
