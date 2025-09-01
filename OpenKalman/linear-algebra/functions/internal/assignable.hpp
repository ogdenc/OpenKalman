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
 * \internal
 * \brief Definition for \ref internal::assignable.
 */

#ifndef OPENKALMAN_ASSIGNABLE_HPP
#define OPENKALMAN_ASSIGNABLE_HPP


namespace OpenKalman::internal
{
  /**
   * \brief Specifies that a type is assignable by the function \ref assign without directly copying elements.
   * \deprecated
   */
  template<typename To, typename From>
#ifdef __cpp_lib_concepts
  concept assignable =
#else
  constexpr bool element_gettable =
#endif
    interface::assign_defined_for<To, To, From> or
    interface::assign_defined_for<To, To, decltype(to_native_matrix<To>(std::declval<From>()))> or
    std::is_assignable_v<To, From> or
    std::is_assignable_v<To, decltype(to_native_matrix<To>(std::declval<From>()))>;


}

#endif
