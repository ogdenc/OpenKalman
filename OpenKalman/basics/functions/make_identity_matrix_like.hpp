/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022 Christopher Lee Ogden <ogden@gatech.edu>
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
   * \brief Make an identity matrix based on an object of a particular library.
   * \tparam T Any matrix or tensor within the relevant library.
   * \tparam Scalar An optional scalar type for the new zero matrix. By default, T's scalar type is used.
   * \param Ds A set of \ref vector_space_descriptor items defining the dimensions of each index.
   */
#ifdef __cpp_concepts
  template<indexible T, scalar_type Scalar = scalar_type_of_t<T>, vector_space_descriptor D, vector_space_descriptor...Ds>
  constexpr identity_matrix auto
#else
  template<typename T, typename Scalar = typename scalar_type_of<T>::type, typename D, typename...Ds, std::enable_if_t<
    indexible<T> and scalar_type<Scalar> and (vector_space_descriptor<D> and ... and vector_space_descriptor<Ds>), int> = 0>
  constexpr auto
#endif
  make_identity_matrix_like(D&& d, Ds&&...ds)
  {
    if constexpr (interface::make_identity_matrix_defined_for<T, Scalar, D&&, Ds&&...>)
      return interface::library_interface<std::decay_t<T>>::template make_identity_matrix<Scalar>(std::forward<D>(d), std::forward<Ds>(ds)...);
    else // Default behavior if interface function not defined:
    {
      auto c = make_constant<T, Scalar, 1>(std::forward<D>(d), Dimensions<1>{});
      return DiagonalMatrix {c, d, ds...};
    }
  }


  /**
   * \overload
   * \brief Make an identity matrix based on the argument, specifying a new scalar type.
   * \tparam T The matrix or array on which the new zero matrix is patterned. It need not be square.
   * \tparam Scalar A scalar type for the new matrix.
   * \return An identity matrix of the same dimensions as T (even if not square).
   */
#ifdef __cpp_concepts
  template<scalar_type Scalar, indexible T>
  constexpr identity_matrix auto
#else
  template<typename Scalar, typename T, std::enable_if_t<scalar_type<Scalar> and indexible<T>, int> = 0>
  constexpr auto
#endif
  make_identity_matrix_like(T&& t)
  {
    if constexpr (identity_matrix<T> and std::is_same_v<Scalar, scalar_type_of_t<T>>) return std::forward<T>(t);
    else return std::apply([](auto&&...ds){ return make_identity_matrix_like<T, Scalar>(ds...); },
      all_vector_space_descriptors(std::forward<T>(t)));
  }


  /**
   * \overload
   * \brief Make an identity matrix based on the argument.
   * \tparam T The matrix or array on which the new zero matrix is patterned.
   * \return An identity matrix of the same dimensions as T (even if not square).
   */
#ifdef __cpp_concepts
  template<indexible T>
  constexpr identity_matrix auto
#else
  template<typename T, std::enable_if_t<indexible<T>, int> = 0>
  constexpr auto
#endif
  make_identity_matrix_like(const T& t)
  {
    return make_identity_matrix_like<scalar_type_of_t<T>>(t);
  }


  /**
   * \overload
   * \brief Make an identity matrix based on T, which has fixed size, specifying a new scalar type.
   * \tparam T The matrix or array on which the new zero matrix is patterned.
   * \tparam Scalar A scalar type for the new matrix. The default is the scalar type of T.
   * \return An identity matrix of the same dimensions as T (even if not square).
   */
#ifdef __cpp_concepts
  template<square_shaped T, scalar_type Scalar = scalar_type_of_t<T>>
  constexpr identity_matrix auto
#else
  template<typename T, typename Scalar = scalar_type_of_t<T>, std::enable_if_t<square_shaped<T> and scalar_type<Scalar>, int> = 0>
  constexpr auto
#endif
  make_identity_matrix_like()
  {
    return std::apply([](auto&&...ds){ return make_identity_matrix_like<T, Scalar>(ds...); },
      all_vector_space_descriptors<T>());
  }

} // namespace OpenKalman

#endif //OPENKALMAN_MAKE_IDENTITY_MATRIX_LIKE_HPP
