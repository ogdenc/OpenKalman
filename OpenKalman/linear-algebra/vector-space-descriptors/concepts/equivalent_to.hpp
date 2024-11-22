/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref equivalent_to.
 */

#ifndef OPENKALMAN_DESCRIPTORS_EQUIVALENT_TO_HPP
#define OPENKALMAN_DESCRIPTORS_EQUIVALENT_TO_HPP

#include "static_vector_space_descriptor.hpp"
#include "maybe_equivalent_to.hpp"


namespace OpenKalman::descriptor
{
  /**
   * \brief Specifies that a set of \ref vector_space_descriptor objects are known at compile time to be equivalent.
   * \details Every descriptor in the set must be equivalent to every other descriptor in the set.
   * Sets of coefficients are equivalent if they are treated functionally the same.
   * - Any coefficient or group of coefficients is equivalent to itself.
   * - StaticDescriptor<As...> is equivalent to StaticDescriptor<Bs...>, if each As is equivalent to its respective Bs.
   * - StaticDescriptor<A> is equivalent to A, and vice versa.
   * \par Example:
   * <code>equivalent_to&lt;Axis, StaticDescriptor&lt;Axis&gt;&gt;</code>
   */
  template<typename...Ts>
#ifdef __cpp_concepts
  concept equivalent_to =
#else
  constexpr bool equivalent_to =
#endif
    (static_vector_space_descriptor<Ts> and ...) and maybe_equivalent_to<Ts...>;


} // namespace OpenKalman::descriptor

#endif //OPENKALMAN_DESCRIPTORS_EQUIVALENT_TO_HPP
