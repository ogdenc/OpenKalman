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
 * \brief Declaration for \untyped_dense_writable_matrix_t.
 */

#ifndef OPENKALMAN_UNTYPED_DENSE_WRITABLE_MATRIX_T_HPP
#define OPENKALMAN_UNTYPED_DENSE_WRITABLE_MATRIX_T_HPP


namespace OpenKalman
{
  /**
   * \brief An alias for a dense, writable matrix, patterned on parameter T.
   * \tparam T A matrix or array from the relevant matrix library.
   * \tparam layout The /ref Layout of the result.
   * \tparam S A scalar type (may or may not be </code>scalar_type_of_t<T></code>.
   * \tparam D Integral values defining the dimensions of the new matrix.
   */
#ifdef __cpp_concepts
  template<indexible T, Layout layout = Layout::none, scalar_type S = scalar_type_of_t<T>, std::integral auto...D> requires
    ((std::is_integral_v<decltype(D)> and D >= 0) and ...) and (layout != Layout::stride)
#else
  template<typename T, Layout layout = Layout::none, typename S = scalar_type_of_t<T>, auto...D>
#endif
  using untyped_dense_writable_matrix_t = dense_writable_matrix_t<T, layout, S, Dimensions<static_cast<const std::size_t>(D)>...>;


} // namespace OpenKalman

#endif //OPENKALMAN_UNTYPED_DENSE_WRITABLE_MATRIX_T_HPP
