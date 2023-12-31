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
   * \tparam T The matrix or tensor of a particular library.
   * \tparam Scalar An optional scalar type for the new zero matrix. By default, T's scalar type is used.
   * \param D A set of \ref vector_space_descriptor defining the dimensions of each index.
   */
#ifdef __cpp_concepts
  template<indexible T, scalar_type Scalar = scalar_type_of_t<T>, vector_space_descriptor D>
  constexpr identity_matrix auto
#else
  template<typename T, typename Scalar = typename scalar_type_of<T>::type, typename D, std::enable_if_t<
    indexible<T> and scalar_type<Scalar> and vector_space_descriptor<D>, int> = 0>
  constexpr auto
#endif
  make_identity_matrix_like(D&& d)
  {
    if constexpr (interface::make_identity_matrix_defined_for<std::decay_t<T>, Scalar, D&&>)
      return interface::library_interface<std::decay_t<T>>::template make_identity_matrix<Scalar>(std::forward<D>(d));
    else // Default behavior if interface function not defined:
      return DiagonalMatrix {make_constant<T, Scalar, 1>(std::forward<D>(d), Dimensions<1>{})};
  }


  /**
   * \overload
   * \brief Make an identity matrix based on the argument, specifying a new scalar type.
   * \tparam T The matrix or array on which the new zero matrix is patterned.
   * \tparam Scalar A scalar type for the new matrix.
   */
#ifdef __cpp_concepts
  template<scalar_type Scalar, square_shaped<Qualification::depends_on_dynamic_shape> T>
  constexpr identity_matrix auto
#else
  template<typename Scalar, typename T, std::enable_if_t<scalar_type<Scalar> and square_shaped<T, Qualification::depends_on_dynamic_shape>, int> = 0>
  constexpr auto
#endif
  make_identity_matrix_like(T&& t)
  {
    if constexpr (identity_matrix<T> and std::is_same_v<Scalar, scalar_type_of_t<T>>)
    {
      return std::forward<T>(t);
    }
    else if constexpr (has_dynamic_dimensions<T>)
    {
      if (get_index_dimension_of<0>(t) != get_index_dimension_of<1>(t)) throw std::invalid_argument {
        "Argument of make_identity_matrix_like must be square; instead it has " +
        std::to_string(get_index_dimension_of<0>(t)) + " rows and " +
        std::to_string(get_index_dimension_of<1>(t)) + " columns"};

      if constexpr (dynamic_dimension<T, 0>)
        return make_identity_matrix_like<T, Scalar>(get_vector_space_descriptor<1>(t));
      else
        return make_identity_matrix_like<T, Scalar>(get_vector_space_descriptor<0>(t));
    }
    else
    {
      return make_identity_matrix_like<T, Scalar>(get_vector_space_descriptor<0>(t));
    }
  }


  /**
   * \overload
   * \brief Make an identity matrix based on the argument.
   * \tparam T The matrix or array on which the new zero matrix is patterned.
   */
#ifdef __cpp_concepts
  template<square_shaped<Qualification::depends_on_dynamic_shape> T>
  constexpr identity_matrix auto
#else
  template<typename T, std::enable_if_t<indexible<T> and square_shaped<T, Qualification::depends_on_dynamic_shape>, int> = 0>
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
    return make_identity_matrix_like<T, Scalar>(Dimensions<index_dimension_of_v<T, 0>>{});
  }

} // namespace OpenKalman

#endif //OPENKALMAN_MAKE_IDENTITY_MATRIX_LIKE_HPP
