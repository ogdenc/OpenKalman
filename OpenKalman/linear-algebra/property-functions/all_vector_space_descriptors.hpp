/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition of \ref all_vector_space_descriptors function.
 */

#ifndef OPENKALMAN_ALL_VECTOR_SPACE_DESCRIPTORS_HPP
#define OPENKALMAN_ALL_VECTOR_SPACE_DESCRIPTORS_HPP

#include "linear-algebra/coordinates/concepts/pattern_collection.hpp"
#include "linear-algebra/coordinates/concepts/pattern_tuple.hpp"
#include "linear-algebra/interfaces/object-traits-defined.hpp"
#include "linear-algebra/concepts/indexible.hpp"
#include "linear-algebra/property-functions/internal/VectorSpaceDescriptorRange.hpp"
#include "linear-algebra/concepts/has_dynamic_dimensions.hpp"
#include "linear-algebra/traits/index_count.hpp"

namespace OpenKalman
{

  namespace detail
  {
    template<typename T, std::size_t...I>
    constexpr auto all_vector_space_descriptors_impl(const T& t, std::index_sequence<I...>)
    {
      return std::tuple {get_vector_space_descriptor<I>(t)...};
    }
  } // namespace detail


  /**
   * \brief Return a collection of \ref coordinate::pattern objects associated with T.
     \details This will be a \ref pattern_collection in the form of a std::tuple or a std::vector.
   */
#ifdef __cpp_concepts
  template<indexible T> requires interface::get_vector_space_descriptor_defined_for<T> 
  constexpr pattern_collection decltype(auto)
#else
  template<typename T, std::enable_if_t<
    indexible<T> and interface::get_vector_space_descriptor_defined_for<T>, int> = 0>
  constexpr decltype(auto) 
#endif
  all_vector_space_descriptors(const T& t)
  {
    if constexpr (index_count_v<T> == dynamic_size)
      return internal::VectorSpaceDescriptorRange<T> {t};
    else 
      return detail::all_vector_space_descriptors_impl(t, std::make_index_sequence<index_count_v<T>>{});
  }


  namespace detail
  {
    template<typename T, std::size_t...I>
    constexpr auto all_vector_space_descriptors_impl(std::index_sequence<I...>)
    {
      return std::tuple {std::decay_t<decltype(get_vector_space_descriptor<I>(std::declval<T>()))>{}...};
    }

  } // namespace detail


  /**
   * \overload
   * \brief Return a collection of \ref fixed_pattern objects associated with T.
   * \details This overload is only enabled if all vector space descriptors are static.
   */
#ifdef __cpp_concepts
  template<indexible T> requires (index_count_v<T> != dynamic_size) and (not has_dynamic_dimensions<T>) 
  constexpr pattern_tuple auto
#else
  template<typename T, std::enable_if_t<indexible<T> and (index_count<T>::value != dynamic_size) and 
    (not has_dynamic_dimensions<T>), int> = 0>
  constexpr auto 
#endif
  all_vector_space_descriptors()
  {
    constexpr std::make_index_sequence<index_count_v<T>> seq;
    return detail::all_vector_space_descriptors_impl<T>(seq);
  }


} // namespace OpenKalman

#endif //OPENKALMAN_ALL_VECTOR_SPACE_DESCRIPTORS_HPP
