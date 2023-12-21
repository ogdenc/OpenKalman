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
 * \brief Definition for \ref element_settable.
 */

#ifndef OPENKALMAN_ELEMENT_SETTABLE_HPP
#define OPENKALMAN_ELEMENT_SETTABLE_HPP


namespace OpenKalman
{
  /**
   * \brief Specifies that a type has elements that can be set with N number of indices (of type std::size_t).
   * \details This concept should include anything for which set_component(...) is properly defined with N std::size_t arguments.
   * \sa set_component
   */
  template<typename T, std::size_t N>
#ifdef __cpp_lib_concepts
  concept element_settable = (N == dynamic_size or N >= index_count_v<T>) and (not std::is_const_v<std::remove_reference_t<T>>) and
    interface::set_component_defined_for<std::decay_t<T>, std::remove_reference_t<T>&, const scalar_type_of_t<T>&, std::array<std::size_t, index_count_v<T>>>;
#else
  constexpr bool element_settable = (N == dynamic_size or N >= index_count_v<T>) and (not std::is_const_v<std::remove_reference_t<T>>) and
    interface::set_component_defined_for<std::decay_t<T>, std::remove_reference_t<T>&, const typename scalar_type_of<T>::type&, std::array<std::size_t, index_count<T>::value>>;
#endif


} // namespace OpenKalman

#endif //OPENKALMAN_ELEMENT_SETTABLE_HPP
