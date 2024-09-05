/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref get_vector_space_descriptor_slice.
 */

#ifndef OPENKALMAN_GET_VECTOR_SPACE_DESCRIPTOR_SLICE_HPP
#define OPENKALMAN_GET_VECTOR_SPACE_DESCRIPTOR_SLICE_HPP

#include <type_traits>


namespace OpenKalman
{
  namespace detail
  {
#ifndef __cpp_concepts
    template<typename T, typename Offset, typename Extent, typename = void>
    struct slice_is_within_range : std::false_type {};

    template<typename T, typename Offset, typename Extent>
    struct slice_is_within_range<T, Offset, Extent, std::enable_if_t<dynamic_index_value<Offset> and dynamic_index_value<Extent>>>
      : std::true_type {};

    template<typename T, typename Offset, typename Extent>
    struct slice_is_within_range<T, Offset, Extent, std::enable_if_t<
        fixed_vector_space_descriptor<T> and static_index_value<Offset> and dynamic_index_value<Extent>>>
      : std::bool_constant<Offset::value >= 0 and Offset::value <= dimension_size_of_v<T>> {};

    template<typename T, typename Offset, typename Extent>
    struct slice_is_within_range<T, Offset, Extent, std::enable_if_t<
        fixed_vector_space_descriptor<T> and dynamic_index_value<Offset> and static_index_value<Extent>>>
      : std::bool_constant<Extent::value >= 0 and Extent::value <= dimension_size_of_v<T>> {};

    template<typename T, typename Offset, typename Extent>
    struct slice_is_within_range<T, Offset, Extent, std::void_t<typename internal::fixed_vector_space_descriptor_slice<T, Offset::value, Extent::value>::type>>
      : std::true_type {};

    template<typename T, typename Offset, typename Extent>
    struct slice_is_within_range<T, Offset, Extent, std::enable_if_t<
        dynamic_vector_space_descriptor<T> and static_index_value<Offset> and dynamic_index_value<Extent>>>
      : std::bool_constant<Offset::value >= 0> {};

    template<typename T, typename Offset, typename Extent>
    struct slice_is_within_range<T, Offset, Extent, std::enable_if_t<
        dynamic_vector_space_descriptor<T> and dynamic_index_value<Offset> and static_index_value<Extent>>>
      : std::bool_constant<Extent::value >= 0> {};
#endif


    template<typename T, std::size_t max_dim, std::size_t position = 0>
    auto get_vector_space_descriptor_slice_impl(std::size_t offset, std::size_t extent, DynamicDescriptor<>&& ds = {})
    {
      if constexpr (vector_space_component_count_v<T> == 0 and position == max_dim)
      {
        return std::move(ds);
      }
      else if (position == offset + extent)
      {
        return std::move(ds);
      }
      else if ((position <= offset or get_dimension_size_of(ds) > 0) and
        (position < offset + extent))
      {
        using Head = std::tuple_element_t<0, internal::split_head_tail_fixed_t<T>>;
        using Tail = std::tuple_element_t<1, internal::split_head_tail_fixed_t<T>>;
        constexpr std::size_t next_position = position + dimension_size_of_v<Head>;
        if (position < offset)
          return get_vector_space_descriptor_slice_impl<Tail, max_dim, next_position>(offset, extent);
        else
          return get_vector_space_descriptor_slice_impl<Tail, max_dim, next_position>(offset, extent, std::move(ds += Head{}));
      }
      else
      {
        throw std::invalid_argument {"get_vector_space_descriptor_slice: invalid arguments"};
      }
    }

  } // namespace detail


  /**
   * \brief Get a slice of \ref vector_space_descriptor T
   * \tparam offset The beginning location of the slice.
   * \tparam extent The size of the slize.
   */
#ifdef __cpp_concepts
  template<vector_space_descriptor T, index_value Offset, index_value Extent> requires
    (dynamic_index_value<Offset> or Offset::value >= 0) and (dynamic_index_value<Extent> or Extent::value >= 0) and
    (dynamic_vector_space_descriptor<T> or
      ((dynamic_index_value<Offset> or Offset::value <= dimension_size_of_v<T>) and
      (dynamic_index_value<Extent> or Extent::value <= dimension_size_of_v<T>) and
      (dynamic_index_value<Offset> or dynamic_index_value<Extent> or requires { typename internal::fixed_vector_space_descriptor_slice_t<T, Offset::value, Extent::value>; })))
#else
  template<typename T, typename Offset, typename Extent, std::enable_if_t<
    vector_space_descriptor<T> and index_value<Offset> and index_value<Extent> and
    detail::slice_is_within_range<T, Offset, Extent>::value, int> = 0>
#endif
  constexpr auto
  get_vector_space_descriptor_slice(T&& t, const Offset& offset, const Extent& extent)
  {
    if constexpr (fixed_vector_space_descriptor<T> and static_index_value<Offset> and static_index_value<Extent>)
    {
      return internal::fixed_vector_space_descriptor_slice_t<T, Offset::value, Extent::value> {};
    }
    else
    {
      if (static_cast<std::size_t>(offset + extent) > get_dimension_size_of(t) or offset < 0 or extent < 0)
        throw std::invalid_argument {"get_vector_space_descriptor_slice: invalid offset and extent"};

      if constexpr (euclidean_vector_space_descriptor<T>) return extent;
      else if constexpr (dynamic_vector_space_descriptor<T>) return std::forward<T>(t).slice(offset, extent);
      else return detail::get_vector_space_descriptor_slice_impl<T, dimension_size_of_v<T>>(offset, extent);
    }
  }


} // namespace OpenKalman


#endif //OPENKALMAN_GET_VECTOR_SPACE_DESCRIPTOR_SLICE_HPP
