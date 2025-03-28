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
 * \brief Declaration for \dense_writable_matrix_t.
 */

#ifndef OPENKALMAN_DENSE_WRITABLE_MATRIX_T_HPP
#define OPENKALMAN_DENSE_WRITABLE_MATRIX_T_HPP


namespace OpenKalman
{
  /**
   * \brief An alias for a dense, writable matrix, patterned on parameter T.
   * \tparam T A matrix or array from the relevant matrix library.
   * \tparam S A scalar type (may or may not be </code>scalar_type_of_t<T></code>.
   * \tparam layout The /ref Layout of the result.
   * \tparam D A \ref pattern_collection defining the new object. This will be derived from T if omitted.
   */
#ifdef __cpp_concepts
  template<indexible T, Layout layout = Layout::none, value::number S = scalar_type_of_t<T>,
    pattern_collection D = decltype(all_vector_space_descriptors(std::declval<T>()))>
      requires (layout != Layout::stride)
#else
  template<typename T, Layout layout = Layout::none, typename S = scalar_type_of_t<T>,
    typename D = decltype(all_vector_space_descriptors(std::declval<T>())), std::enable_if_t<
      indexible<T> and value::number<S> and pattern_collection<D>, int> = 0>
#endif
  using dense_writable_matrix_t = std::decay_t<decltype(make_dense_object<T, layout, S>(std::declval<D>()))>;



} // namespace OpenKalman


#endif //OPENKALMAN_DENSE_WRITABLE_MATRIX_T_HPP
