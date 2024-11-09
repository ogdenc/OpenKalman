/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definitions for \ref make_identity_matrix_like.
 */

#ifndef OPENKALMAN_MAKE_IDENTITY_MATRIX_LIKE_HPP
#define OPENKALMAN_MAKE_IDENTITY_MATRIX_LIKE_HPP

namespace OpenKalman
{
  /**
   * \brief Make an \ref identity_matrix with a particular set of dimensions.
   * \tparam T Any matrix or tensor within the relevant library.
   * \tparam Scalar An optional scalar type for the new zero matrix. By default, T's scalar type is used.
   * \param Ds A set of \ref vector_space_descriptor items defining the dimensions of each index.
   */
#ifdef __cpp_concepts
  template<indexible T, scalar_type Scalar = scalar_type_of_t<T>, vector_space_descriptor_collection Descriptors>
  constexpr identity_matrix auto
#else
  template<typename T, typename Scalar = typename scalar_type_of<T>::type, typename Descriptors, std::enable_if_t<
    indexible<T> and scalar_type<Scalar> and vector_space_descriptor_collection<Descriptors>, int> = 0>
  constexpr auto
#endif
  make_identity_matrix_like(Descriptors&& descriptors)
  {
    if constexpr (interface::make_identity_matrix_defined_for<T, Scalar, Descriptors>)
    {
      using Traits = interface::library_interface<std::decay_t<T>>;
      if constexpr (euclidean_vector_space_descriptor_collection<Descriptors>)
        return Traits::template make_identity_matrix<Scalar>(std::forward<Descriptors>(descriptors));
      else
        return make_vector_space_adapter(
          Traits::template make_identity_matrix<Scalar>(internal::to_euclidean_vector_space_descriptor_collection(descriptors)),
          std::forward<Descriptors>(descriptors));
    }
    else
    {
      return internal::make_constant_diagonal_from_descriptors<T>(
        values::ScalarConstant<Scalar, 1>{}, std::forward<Descriptors>(descriptors));
    }
  }


  /**
   * \overload
   * \brief Make an identity matrix with the same size and shape as an argument, specifying a new scalar type.
   * \tparam Arg The matrix or array on which the new identity matrix is patterned. It need not be square.
   * \tparam Scalar A scalar type for the new matrix.
   * \return An identity matrix of the same dimensions as Arg (even if not square).
   */
#ifdef __cpp_concepts
  template<scalar_type Scalar, indexible Arg>
  constexpr identity_matrix auto
#else
  template<typename Scalar, typename Arg, std::enable_if_t<scalar_type<Scalar> and indexible<Arg>, int> = 0>
  constexpr auto
#endif
  make_identity_matrix_like(Arg&& arg)
  {
    if constexpr (identity_matrix<Arg> and std::is_same_v<Scalar, scalar_type_of_t<Arg>>) return std::forward<Arg>(arg);
    else return make_identity_matrix_like<Arg, Scalar>(all_vector_space_descriptors(std::forward<Arg>(arg)));
  }


  /**
   * \overload
   * \brief Make an identity matrix with the same size and shape as an argument.
   * \tparam Arg The matrix or array on which the new zero matrix is patterned.
   * \return An identity matrix of the same dimensions as Arg (even if not square).
   */
#ifdef __cpp_concepts
  template<indexible Arg>
  constexpr identity_matrix auto
#else
  template<typename Arg, std::enable_if_t<indexible<Arg>, int> = 0>
  constexpr auto
#endif
  make_identity_matrix_like(Arg&& arg)
  {
    return make_identity_matrix_like<scalar_type_of_t<Arg>>(std::forward<Arg>(arg));
  }


} // namespace OpenKalman

#endif //OPENKALMAN_MAKE_IDENTITY_MATRIX_LIKE_HPP
