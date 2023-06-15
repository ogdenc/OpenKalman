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
#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename Scalar, typename D, typename = void>
    struct make_identity_matrix_trait_defined: std::false_type {};

    template<typename T, typename Scalar, typename D>
    struct make_identity_matrix_trait_defined<T, Scalar, D, std::void_t<
      decltype(interface::SingleConstantDiagonalMatrixTraits<T, Scalar>::make_identity_matrix(std::declval<D&&>()))>>
      : std::true_type {};
  }
#endif


  /**
   * \brief Make an identity matrix based on an object of a particular library.
   * \tparam T The matrix or tensor of a particular library.
   * \tparam Scalar An optional scalar type for the new zero matrix. By default, T's scalar type is used.
   * \param D An \ref index_descriptor "index descriptor" defining the dimensions of each index.
   */
#ifdef __cpp_concepts
  template<indexible T, scalar_type Scalar = scalar_type_of_t<T>, index_descriptor D>
  constexpr identity_matrix auto
#else
  template<typename T, typename Scalar = typename scalar_type_of<T>::type, typename D, std::enable_if_t<
    indexible<T> and scalar_type<Scalar> and index_descriptor<D>, int> = 0>
  constexpr auto
#endif
  make_identity_matrix_like(D&& d)
  {
    using Td = std::decay_t<T>;
#ifdef __cpp_concepts
    if constexpr (requires (D&& d) { interface::SingleConstantDiagonalMatrixTraits<Td, Scalar>::make_identity_matrix(std::forward<D>(d)); })
#else
    if constexpr (detail::make_identity_matrix_trait_defined<Td, Scalar, D>::value)
#endif
    {
      return interface::SingleConstantDiagonalMatrixTraits<Td, Scalar>::make_identity_matrix(std::forward<D>(d));
    }
    else
    {
      // Default behavior if interface function not defined:
      return DiagonalMatrix {make_constant_matrix_like<Td, Scalar, 1>(std::forward<D>(d), Dimensions<1>{})};
    }
  }


  /**
   * \overload
   * \brief Make an identity matrix based on the argument, specifying a new scalar type.
   * \tparam T The matrix or array on which the new zero matrix is patterned.
   * \tparam Scalar A scalar type for the new matrix.
   */
#ifdef __cpp_concepts
  template<scalar_type Scalar, square_matrix<Likelihood::maybe> T>
  constexpr identity_matrix auto
#else
  template<typename Scalar, typename T, std::enable_if_t<scalar_type<Scalar> and square_matrix<T, Likelihood::maybe>, int> = 0>
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
        return make_identity_matrix_like<T, Scalar>(get_index_descriptor<1>(t));
      else
        return make_identity_matrix_like<T, Scalar>(get_index_descriptor<0>(t));
    }
    else
    {
      return make_identity_matrix_like<T, Scalar>(get_index_descriptor<0>(t));
    }
  }


  /**
   * \overload
   * \brief Make an identity matrix based on the argument.
   * \tparam T The matrix or array on which the new zero matrix is patterned.
   */
#ifdef __cpp_concepts
  template<square_matrix<Likelihood::maybe> T>
  constexpr identity_matrix auto
#else
  template<typename T, std::enable_if_t<indexible<T> and square_matrix<T, Likelihood::maybe>, int> = 0>
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
  template<square_matrix T, scalar_type Scalar = scalar_type_of_t<T>>
  constexpr identity_matrix auto
#else
  template<typename T, typename Scalar = scalar_type_of_t<T>, std::enable_if_t<square_matrix<T> and scalar_type<Scalar>, int> = 0>
  constexpr auto
#endif
  make_identity_matrix_like()
  {
    return make_identity_matrix_like<T, Scalar>(Dimensions<index_dimension_of_v<T, 0>>{});
  }

} // namespace OpenKalman

#endif //OPENKALMAN_MAKE_IDENTITY_MATRIX_LIKE_HPP