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
 * \brief Definition for \ref element_gettable.
 */

#ifndef OPENKALMAN_ELEMENT_GETTABLE_HPP
#define OPENKALMAN_ELEMENT_GETTABLE_HPP


namespace OpenKalman
{
  /**
   * \brief Specifies that a type has components addressable by N indices.
   * \details This concept should include anything for which get_component(...) is properly defined with N std::size_t arguments.
   * \sa get_component
   * \deprecated
   */
  template<typename T, std::size_t N>
#ifdef __cpp_lib_concepts
  concept element_gettable = (N == dynamic_size or N >= index_count_v<T>) and
    interface::get_component_defined_for<T, T, std::array<std::size_t, index_count_v<T>>>;
#else
  constexpr bool element_gettable = (N == dynamic_size or N >= index_count_v<T>) and
    interface::get_component_defined_for<T, T, std::array<std::size_t, index_count<T>::value>>;
#endif


}

#endif
