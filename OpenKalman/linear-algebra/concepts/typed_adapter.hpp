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
 * \brief Declaration for \typed_adapter.
 */

#ifndef OPENKALMAN_TYPED_ADAPTER_HPP
#define OPENKALMAN_TYPED_ADAPTER_HPP


namespace OpenKalman
{
  // --------------- //
  //  typed_adapter  //
  // --------------- //

  /**
   * \brief Specifies that T is a typed adapter expression.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept typed_adapter =
#else
  constexpr bool typed_adapter =
#endif
    typed_matrix<T> or covariance<T> or euclidean_expr<T>;

}


#endif
