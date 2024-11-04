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
 * \brief Definition for \ref writable_by_component.
 */

#ifndef OPENKALMAN_WRITABLE_BY_COMPONENT_HPP
#define OPENKALMAN_WRITABLE_BY_COMPONENT_HPP


namespace OpenKalman
{
  /**
   * \brief Specifies that a type has components that can be set with Indices (an std::ranges::input_range) of type std::size_t.
   * \details If T satisfies this concept, then set_component(...) is available with Indices.
   * \sa set_component
   */
  template<typename T, typename Indices =
    std::conditional_t<index_count_v<T> == dynamic_size, std::array<std::size_t, index_count_v<T>>, std::vector<std::size_t>>>
#ifdef __cpp_lib_concepts
  concept writable_by_component = 
    indexible<T> and static_range_size<Indices, T> and 
    (not std::is_const_v<std::remove_reference_t<T>>) and (not empty_object<T>) and
    interface::set_component_defined_for<
      T, std::add_lvalue_reference_t<T>, const scalar_type_of_t<T>&, const Indices&>;
#else
  constexpr bool writable_by_component = 
    indexible<T> and static_range_size<Indices, T> and 
    (not std::is_const_v<std::remove_reference_t<T>>) and (not empty_object<T>) and
    interface::set_component_defined_for<
      T, std::add_lvalue_reference_t<T>, const typename scalar_type_of<T>::type&, const Indices&>;
#endif


} // namespace OpenKalman

#endif //OPENKALMAN_WRITABLE_BY_COMPONENT_HPP
