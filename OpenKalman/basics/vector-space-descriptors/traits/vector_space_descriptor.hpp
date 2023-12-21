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
 * \brief Definition for \ref vector_space_descriptor.
 */

#ifndef OPENKALMAN_VECTOR_SPACE_DESCRIPTOR_HPP
#define OPENKALMAN_VECTOR_SPACE_DESCRIPTOR_HPP


namespace OpenKalman
{
  /**
   * \brief An object describing the type of (vector) space associated with a tensor index.
   * \details Such an object is a trait defining the number of dimensions and whether each dimension is modular.
   * This includes anything that is either a \ref fixed_vector_space_descriptor or a \ref dynamic_vector_space_descriptor.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept vector_space_descriptor =
#else
  constexpr bool vector_space_descriptor =
#endif
    fixed_vector_space_descriptor<T> or dynamic_vector_space_descriptor<T>;


} // namespace OpenKalman

#endif //OPENKALMAN_VECTOR_SPACE_DESCRIPTOR_HPP
