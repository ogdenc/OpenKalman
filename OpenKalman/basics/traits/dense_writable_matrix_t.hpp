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
 * \brief Declaration for \dense_writable_matrix_t.
 */

#ifndef OPENKALMAN_DENSE_WRITABLE_MATRIX_T_HPP
#define OPENKALMAN_DENSE_WRITABLE_MATRIX_T_HPP


namespace OpenKalman
{
  namespace detail
  {
    template<typename T, Layout layout, typename Scalar, typename...D>
    struct dense_writable_matrix_impl
    {
      using type = std::decay_t<decltype(make_dense_object<T, layout, Scalar>(std::declval<D>()...))>;
    };


    template<typename T, Layout layout, typename Scalar>
    struct dense_writable_matrix_impl<T, layout, Scalar>
    {
      using type = std::decay_t<decltype(make_dense_object<layout, Scalar>(std::declval<T>()))>;
    };
  }


  /**
   * \brief An alias for a dense, writable matrix, patterned on parameter T.
   * \tparam T A matrix or array from the relevant matrix library.
   * \tparam S A scalar type (may or may not be </code>scalar_type_of_t<T></code>.
   * \tparam layout The /ref Layout of the result.
   * \tparam D \ref vector_space_descriptor objects defining the dimensions of the new matrix.
   * \todo Create typed Matrix if Ds are typed.
   */
#ifdef __cpp_concepts
  template<indexible T, Layout layout = Layout::none, scalar_type S = scalar_type_of_t<T>, vector_space_descriptor...D>
    requires (layout != Layout::stride)
#else
  template<typename T, Layout layout = Layout::none, typename S = scalar_type_of_t<T>, typename...D>
#endif
  using dense_writable_matrix_t = typename detail::dense_writable_matrix_impl<T, layout, std::decay_t<S>, D...>::type;



} // namespace OpenKalman


#endif //OPENKALMAN_DENSE_WRITABLE_MATRIX_T_HPP
