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
 * \brief Declaration for \untyped_adapter.
 */

#ifndef OPENKALMAN_UNTYPED_ADAPTER_HPP
#define OPENKALMAN_UNTYPED_ADAPTER_HPP


namespace OpenKalman
{
  /**
   * \brief Specifies that T is an untyped adapter expression.
   * \details Untyped adapter expressions are generally used whenever the native matrix library does not have an
   * important built-in matrix type, such as a diagonal matrix, a triangular matrix, or a hermitian matrix.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept untyped_adapter =
#else
  constexpr bool untyped_adapter =
#endif
    eigen_diagonal_expr<T> or eigen_self_adjoint_expr<T> or eigen_triangular_expr<T>;


} // namespace OpenKalman


#endif //OPENKALMAN_UNTYPED_ADAPTER_HPP
