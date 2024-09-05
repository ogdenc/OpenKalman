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
 * \internal
 * \file
 * \brief Definition for \ref fixed_vector_space_descriptor_slice.
 */

#ifndef OPENKALMAN_VECTOR_SPACE_DESCRIPTOR_SLICE_HPP
#define OPENKALMAN_VECTOR_SPACE_DESCRIPTOR_SLICE_HPP

#include <type_traits>


namespace OpenKalman::internal
{
  namespace detail
  {
#ifdef __cpp_concepts
    template<typename T, std::size_t offset, std::size_t extent, std::size_t position = 0, typename Result = FixedDescriptor<>>
#else
    template<typename T, std::size_t offset, std::size_t extent, std::size_t position = 0, typename Result = FixedDescriptor<>, typename = void>
#endif
    struct fixed_descriptor_slice_impl {};


#ifdef __cpp_concepts
    template<typename T, std::size_t offset, std::size_t extent, std::size_t position, typename...Ds> requires
      (position <= offset or sizeof...(Ds) > 0) and (position < offset + extent)
    struct fixed_descriptor_slice_impl<T, offset, extent, position, FixedDescriptor<Ds...>>
#else
    template<typename T, std::size_t offset, std::size_t extent, std::size_t position, typename...Ds>
    struct fixed_descriptor_slice_impl<T, offset, extent, position, FixedDescriptor<Ds...>, std::enable_if_t<
      (position <= offset or sizeof...(Ds) > 0) and (position < offset + extent)>>
#endif
    {
      using type = typename fixed_descriptor_slice_impl<
        std::tuple_element_t<1, internal::split_head_tail_fixed_t<T>>,
        offset,
        extent,
        position + dimension_size_of_v<std::tuple_element_t<0, internal::split_head_tail_fixed_t<T>>>,
        std::conditional_t<
          (position < offset),
          FixedDescriptor<>,
          FixedDescriptor<Ds..., std::tuple_element_t<0, internal::split_head_tail_fixed_t<T>>>>>::type;
    };


#ifdef __cpp_concepts
    template<typename T, std::size_t offset, std::size_t extent, std::size_t position, typename...Ds> requires
      (position == offset + extent)
    struct fixed_descriptor_slice_impl<T, offset, extent, position, FixedDescriptor<Ds...>>
#else
    template<typename T, std::size_t offset, std::size_t extent, std::size_t position, typename...Ds>
    struct fixed_descriptor_slice_impl<T, offset, extent, position, FixedDescriptor<Ds...>, std::enable_if_t<
      (position == offset + extent)>>
#endif
      {
        using type = FixedDescriptor<Ds...>;
      };

  } // namespace detail


  /**
   * \brief Get a slice of \ref fixed_vector_space_descriptor T
   * \details If a slice is valid, member <code>type</code> will be an alias for that type.
   * \tparam offset The beginning location of the slice.
   * \tparam extent The size of the slice.
   */
#ifdef __cpp_concepts
  template<fixed_vector_space_descriptor T, std::size_t offset, std::size_t extent>
#else
  template<typename T, std::size_t offset, std::size_t extent, typename = void>
#endif
  struct fixed_vector_space_descriptor_slice;


#ifdef __cpp_concepts
  template<fixed_vector_space_descriptor T, std::size_t offset, std::size_t extent> requires
    (offset + extent <= dimension_size_of_v<T>) and euclidean_vector_space_descriptor<T>
  struct fixed_vector_space_descriptor_slice<T, offset, extent>
#else
  template<typename T, std::size_t offset, std::size_t extent>
  struct fixed_vector_space_descriptor_slice<T, offset, extent, std::enable_if_t<fixed_vector_space_descriptor<T> and
    (offset + extent <= dimension_size_of_v<T>) and euclidean_vector_space_descriptor<T>>>
#endif
  {
    using type = Dimensions<extent>;
  };


#ifdef __cpp_concepts
  template<fixed_vector_space_descriptor T, std::size_t offset, std::size_t extent> requires
    (offset + extent <= dimension_size_of_v<T>) and (not euclidean_vector_space_descriptor<T>)
  struct fixed_vector_space_descriptor_slice<T, offset, extent>
#else
  template<typename T, std::size_t offset, std::size_t extent>
  struct fixed_vector_space_descriptor_slice<T, offset, extent, std::enable_if_t<fixed_vector_space_descriptor<T> and
    (offset + extent <= dimension_size_of_v<T>) and (not euclidean_vector_space_descriptor<T>)>>
#endif
  {
    using type = typename detail::fixed_descriptor_slice_impl<T, offset, extent>::type;
  };


  /**
   * \brief Helper template for \ref fixed_vector_space_descriptor_slice.
   */
#ifdef __cpp_concepts
  template<fixed_vector_space_descriptor T, std::size_t offset, std::size_t extent> requires
    (offset + extent <= dimension_size_of_v<T>)
#else
  template<typename T, std::size_t offset, std::size_t extent>
#endif
  using fixed_vector_space_descriptor_slice_t = typename fixed_vector_space_descriptor_slice<T, offset, extent>::type;


} // namespace OpenKalman::internal


#endif //OPENKALMAN_VECTOR_SPACE_DESCRIPTOR_SLICE_HPP
