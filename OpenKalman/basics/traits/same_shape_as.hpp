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
 * \brief Definition for \ref same_shape_as.
 */

#ifndef OPENKALMAN_SAME_SHAPE_AS_HPP
#define OPENKALMAN_SAME_SHAPE_AS_HPP


namespace OpenKalman
{
  /**
   * \brief Specifies that T has the same dimensions and vector-space types as Ts.
   * \details Two dimensions are considered the same if their \ref vector_space_descriptor are \ref equivalent_to "equivalent".
   * \sa maybe_same_shape_as
   * \sa same_shape
   */
  template<typename...Ts>
#ifdef __cpp_concepts
  concept same_shape_as =
#else
  constexpr bool same_shape_as =
#endif
    maybe_same_shape_as<Ts...> and ((not has_dynamic_dimensions<Ts>) and ...);


} // namespace OpenKalman

#endif //OPENKALMAN_SAME_SHAPE_AS_HPP
