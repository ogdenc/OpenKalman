/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref has_nested_object.
 */

#ifndef OPENKALMAN_HAS_NESTED_OBJECT_HPP
#define OPENKALMAN_HAS_NESTED_OBJECT_HPP


namespace OpenKalman
{
  /**
   * \brief A matrix that has a nested matrix, if it is a wrapper type.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept has_nested_object =
#else
  constexpr bool has_nested_object =
#endif
    interface::nested_object_defined_for<T>;


}

#endif
